#!/usr/bin/env python3
"""A Python function"""
def schools_by_topic(mongo_collection, topic):
    """Return the list of school documents that have a specific topic."""
    return list(mongo_collection.find({"topics": topic}))
