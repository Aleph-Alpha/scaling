# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: text_image_dataset.proto
# Protobuf Python Version: 4.25.3
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder  # type: ignore
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x18text_image_dataset.proto\"=\n\x13PromptImageLocation\x12\x13\n\x0bstart_index\x18\x01 \x01(\x05\x12\x11\n\tend_index\x18\x02 \x01(\x05\"\xb0\x01\n\x10TextImageExample\x12\x18\n\x10input_token_list\x18\x01 \x03(\x05\x12\x19\n\x11target_token_list\x18\x02 \x03(\x05\x12\x16\n\x0eloss_mask_list\x18\x03 \x03(\x08\x12\x19\n\x11prompt_image_data\x18\x04 \x03(\x0c\x12\x34\n\x16prompt_image_locations\x18\x05 \x03(\x0b\x32\x14.PromptImageLocationb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'text_image_dataset_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:  # type: ignore
  DESCRIPTOR._options = None
  _globals['_PROMPTIMAGELOCATION']._serialized_start=28
  _globals['_PROMPTIMAGELOCATION']._serialized_end=89
  _globals['_TEXTIMAGEEXAMPLE']._serialized_start=92
  _globals['_TEXTIMAGEEXAMPLE']._serialized_end=268
# @@protoc_insertion_point(module_scope)