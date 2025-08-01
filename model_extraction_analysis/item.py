#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from dataclasses import dataclass, astuple, field
#from typing import Any

@dataclass
class apk_item:
    '''class for an apk item'''
    apk_hash: str = ''
    apk_name: str = ''
    apk_path: str = ''
    apk_feature: str = ' '
    #pattern: str = ''
    #framework: str = ''
    #model_format: str = ''
    label: bool = False
    
    def _tuple(self):
        return astuple(self)
    
@dataclass
class ai_apk_item:
    '''class for an ai apk item'''
    apk_hash: str = ''
    apk_name: str = ''
    ori_path: str = ''
    dec_path: str = ''
    framework: list = field(default_factory=list)
    model_format: list = field(default_factory=list)
    magic_string: list = field(default_factory=list)
    
    def _tuple(self):
        return astuple(self)

@dataclass
class model_item:
    '''class for an on-device model item'''
    model_hash: str    = ''
    model_name: str    = ''
    ori_apk_hash: str  = ''
    model_path: str    = ''
    model_type: str    = ''
    quant_level: str   = ''
    framework: str     = ''
    filesize: str      = ''
    task: str          = ''
    backbone: str      = ''
    input_layer: str   = ''
    input_size: str    = ''
    input_type: str    = 'unknown'
    preprocess_param: str = 'unknown'
    output_layer: str  = ''
    output_size: str   = ''
    output_label: str  = 'unknown'
    attack_result: str = ''

    def deserialize(self, t):
        self.model_hash   = t[0]
        self.model_name   = t[1]
        self.ori_apk_hash = t[2]
        self.model_path   = t[3]
        self.model_type   = t[4]
        self.quant_level  = t[5]
        self.framework    = t[6]
        self.filesize     = t[7]
        self.task         = t[8]
        self.backbone     = t[9]
        self.input_layer  = t[10]
        self.input_size   = t[11]
        self.input_type   = t[12]
        self.preprocess_param = t[13]
        self.output_layer  = t[14]
        self.output_size   = t[15]
        self.output_label  = t[16]
        self.attack_result = t[17]
        return self

    def _tuple(self):
        return astuple(self)