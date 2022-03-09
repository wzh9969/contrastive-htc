# Copy from https://github.com/ductri/reuters_loader

import logging
from pathlib import Path
import tarfile
import gzip
import shutil
from urllib import request
from collections import defaultdict
import argparse
import xml.etree.ElementTree as ET
import xml

import pandas as pd


def might_extract_tar(path):
    path = Path(path)
    dir_name = '.'.join(path.name.split('.')[:-2])
    dir_output = path.parent / dir_name
    if not dir_output.exists():
        if path.exists():
            tf = tarfile.open(str(path))
            tf.extractall(path.parent)
        else:
            logging.error('File %s is required. \n', path.name)


def might_extract_gz(path):
    path = Path(path)
    file_output_name = '.'.join(path.name.split('.')[:-1])
    file_name = path.name
    if not (path.parent / file_output_name).exists():
        logging.info('Extracting %s ...\n', file_name)

        with gzip.open(str(path), 'rb') as f_in:
            with open(str(path.parent / file_output_name), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


def might_download_file(url):
    file_name = url.split('/')[-1]
    file = ROOT / file_name
    if not file.exists():
        logging.info('File %s does not exist. Downloading ...\n', file_name)
        file_data = request.urlopen(url)
        data_to_write = file_data.read()

        with file.open('wb') as f:
            f.write(data_to_write)
    else:
        logging.info('File %s already existed.\n', file_name)


def get_doc_ids_v2():
    file = ROOT / 'rcv1v2-ids.dat'
    with file.open('rt', encoding='ascii') as i_f:
        doc_ids = i_f.readlines()
    doc_ids = [item[:-1] for item in doc_ids]
    logging.info('There are %s docs in RCV1-v2\n', len(doc_ids))
    return doc_ids


def get_doc_topics_mapping():
    file = ROOT / 'rcv1-v2.topics.qrels'
    with file.open('rt', encoding='ascii') as i_f:
        lines = i_f.readlines()
    lines = [item[:-1] for item in lines]
    doc_topics = defaultdict(list)
    for item in lines:
        topic, doc_id, _ = item.split()
        doc_topics[doc_id].append(topic)
    logging.info('Mapping dictionary contains %s docs\n', len(doc_topics))
    return doc_topics


def get_topic_desc():
    file = ROOT / 'rcv1' / 'codes' / 'topic_codes.txt'
    with file.open('rt', encoding='ascii') as i_f:
        lines = i_f.readlines()
    lines = [item[:-1] for item in lines if len(item) > 1][2:]
    topic_desc = [item.split('\t') for item in lines]
    topic_desc = {item[0]: item[1] for item in topic_desc}

    logging.info('There are totally %s topics\n', len(topic_desc))
    return topic_desc


def fetch_docs(doc_ids):
    all_path_docs = list(ROOT.glob('rcv1/199*/*'))
    docid2topics = get_doc_topics_mapping()

    docid2path = {p.name[:-10]: p for p in all_path_docs}
    for idx, doc_id in enumerate(doc_ids):
        # text = docid2path[doc_id].open('rt', encoding='iso-8859-1').read()
        tree = ET.parse(str(docid2path[doc_id]))
        root = tree.getroot()
        text = xml.etree.ElementTree.tostring(root, encoding='unicode')
        label = docid2topics[doc_id]
        if idx % 100000 == 0:
            logging.info('Fetched %s/%s docs', idx, len(docs_ids))
        yield doc_id, text, label, str(docid2path[doc_id])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', type=str, help='Absolute path to directory which contains rcv1.tar.xz')
    args = parser.parse_args()

    ROOT = Path(args.root_dir)

    logging.info('Downloading rcv1v2-ids.dat.gz')
    might_download_file(
        'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a07-rcv1-doc-ids/rcv1v2-ids.dat.gz')

    logging.info('Downloading rcv1-v2.topics.qrels.gz')
    might_download_file(
        'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a08-topic-qrels/rcv1-v2.topics.qrels.gz')

    might_extract_gz(ROOT / 'rcv1v2-ids.dat.gz')

    might_extract_gz(ROOT / 'rcv1-v2.topics.qrels.gz')

    logging.info('Extracting main dataset from rcv1.tar.xz')
    might_extract_tar(ROOT / 'rcv1.tar.xz')

    docs_ids = get_doc_ids_v2()

    docs = list(fetch_docs(docs_ids))

    pd.DataFrame(docs, columns=['id', 'text', 'topics', 'path']).to_csv(str(ROOT / 'rcv1_v2.csv'), index=None)
    logging.info('Exported data to %s', str(ROOT / 'rcv1_v2.csv'))

    topic_descriptions = get_topic_desc()
    topic_descriptions = [{'topic_code': k, 'topic_desc': v} for k, v in topic_descriptions.items()]
    pd.DataFrame(topic_descriptions).to_csv(str(ROOT / 'rcv1_v2_topics_desc.csv'), index=None)
    logging.info('Exported topics descriptions to %s', str(ROOT / 'rcv1_v2_topics_desc.csv'))