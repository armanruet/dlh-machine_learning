#!/usr/bin/env python3
"""function for ists all documents in a collection"""
def list_all(mongo_collection):
    """defining the function"""
    return list(mongo_collection.find())
