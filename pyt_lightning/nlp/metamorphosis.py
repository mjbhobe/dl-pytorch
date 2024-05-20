#!/usr/bin/env python
# -*- coding: utf-8 -*-

filename = "metamorphosis_cleaned.txt"
with open(filename, "r") as f:
    text = f.read()

# split text into words on whitespace
words = text.split()
print(" ".join(words[:1000]))
