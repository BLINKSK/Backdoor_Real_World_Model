#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from utils import *

import os
import sys
import logging
import argparse
import filetype
import sqlite3 as sql
from datetime import datetime

def create_db(db_name, schema):
    try:
        with sql.connect(db_name) as con:
            cursor = con.cursor()
            cursor.execute(schema)
    except sql.Error as e:
        logger.error(f"[ERROR]: Create table in {db_name} failed! Detail: {e}")
    con.close()

# executemany
def insert_db(db_name, command, items, single=False):
    try:
        with sql.connect(db_name) as con:
            cursor = con.cursor()
            if single:
                cursor.execute(command, items)
            else:
                cursor.executemany(command, items)
    except sql.Error as e:
        logger.error(f"[ERROR] Inert table failed! {e}")
    except Exception as e:
        logger.error(e)

def update_db(db_name, command, items):
    try:
        with sql.connect(db_name) as con:
            cursor = con.cursor()
            for i in items:
                cursor.execute(command, i)
    except sql.Error as e:
        logger.error(f"[ERROR] Update table failed! {e}")
    except Exception as e:
        logger.error(e)

def query_db(db_name, command, items=None, is_raw=False):
    result = []
    try:
        with sql.connect(db_name) as con:
            if is_raw: con.row_factory = lambda cursor, row: row[0]
            cursor = con.cursor()
            if items == None:
                cursor.execute(command)
                result = cursor.fetchall()
            else:
                # cannot use executemany
                for i in items:
                    cursor.execute(command, i)
                    result.append((i, cursor.fetchall()))
    except sql.Error as e:
        logger.error(f"[ERROR] Query table failed! {e}")
    except Exception as e:
        logger.error(e)
    return result

def read_raw(path):
    """traversing the raw data path, collect apk files."""
    items = []
    for roots, dirs, files in os.walk(path):
        for filename in files:
            file_path = os.path.join(roots, filename)
            kind = filetype.guess(file_path)
            if kind is None:
                logger.warning(f"[DEBUG] {file_path} is [Unknow File Type]") 
            elif kind.extension == 'zip':
                print(file_path)
                i = parse_apk(file_path)
                if i != None:
                    logger.warning("[PARSE_SUC] " + str(i))
                    items.append(i)
            else:
                logger.warning(f"[DEBUG] {file_path} is a {kind.extension} file.")
    return items

def collect_test():
    collect_schema = "create table if not exists apk \
                            (hash text primary key,\
                            name text NOT NULL,\
                            path text NOT NULL,\
                            feature text NOT NULL,\
                            label boolean NOT NULL\
                            )"
    create_db(args.DB_NAME, collect_schema)
    items = read_raw(args.RAW_DATA_PATH)
    insert_cmd = f"insert or replace into apk values (?, ?, ?, ?, ?)"
    insert_db(args.DB_NAME, insert_cmd, items)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Collect apk files from the dir.')
    parser.add_argument('--DB_NAME', help='Dst database path')
    parser.add_argument('--RAW_DATA_PATH', help='Raw data path, MUST USE THE FULL NAME PATH!')
    args = parser.parse_args()

    log_name = os.path.dirname(args.DB_NAME) + '/log'
    os.makedirs(log_name, exist_ok=True)
    logging.basicConfig(filename=log_name + '/' + datetime.now().strftime(f'{os.path.basename(__file__)}_%Y_%m_%d_%H_%M.log'),
                    filemode='a',
                    level=logging.WARNING,
                    format='%(asctime)s  %(message)s',
                    datefmt='%d %b %Y %H:%M:%S')
    logger = logging.getLogger('dev')
    
    collect_test()


