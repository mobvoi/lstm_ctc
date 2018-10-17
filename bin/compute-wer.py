# Copyright 2018 Mobvoi Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.

#!/usr/bin/python2

import argparse
import unicodedata
import sys


def parse_text_line(line):
    res = list()
    word = ''
    line = line + ' '
    res.append(line[0:line.find(' ')])
    line = line[line.find(' ')+1:]
    prev_category = 0
    for char in line:
        category = unicodedata.category(char)
        if category == 'Zs':
            category = 1  # spaces like ' '.
        elif category == 'Lu' or category == 'Ll' or category == 'Nd':
            category = 2  # alphabet/numeric including upper/lower cases.
        else:
            category = 3  # Chinese characters.

        if category == 1 or prev_category != category:
            if word:
                res.append(word)
            word = '' if char == ' ' else char
        else:
            word = word + char
        prev_category = category
    return res


def normalize_tokens(tokens, to_character = False, to_lower = False):
    res = list()
    for tok in tokens:
        if unicodedata.category(tok[0]) == "Lo" and to_character:
            for c in tok:
                res.append(c)
        elif unicodedata.category(tok[0]) != "Lo" and to_lower:
            res.append(tok.lower())
        else:
            res.append(tok)
    return res


def read_text_file(fn, to_character = False, to_lower = False):
    res = dict()
    for line in open(fn, 'r'):
        line = line.decode('utf-8').strip()
        tokens = parse_text_line(line)
        fid = tokens[0]
        words = normalize_tokens(tokens[1:], to_character, to_lower)
        res[fid] = words
    return res


# global matrix for distances and backtrace pointers
dist = [[]]
bptr = [[]]


def allocate_global_variables(m, n):
    while len(dist) < (m + 1):
        dist.append([ 0 ] * (n + 1))
    for i in xrange(m + 1):
        if len(dist[i]) < (n + 1):
            dist[i] = [ 0 ] * (n + 1)
    while len(bptr) < (m + 1):
        bptr.append([ 0 ] * (n + 1))
    for i in xrange((m + 1)):
        if len(bptr[i]) < (n + 1):
            bptr[i] = [ 0 ] * (n + 1)


def initialize_global_variables(m, n):
    allocate_global_variables(m, n)
    for i in xrange(m + 1):
        dist[i][0] = i
    for j in xrange(n + 1):
        dist[0][j] = j
    for i in xrange(m + 1):
        bptr[i][0] = 1  # 1 means along column-direction.
    for j in xrange(n + 1):
        bptr[0][j] = 2  # 2 means alogn row-direction.
    dist[0][0] = 0
    bptr[0][0] = 0  # 0 means no direction for backtracing.


def compute_distance_backtrace(ref, rec):
    m = len(ref)
    n = len(rec)
    allocate_global_variables(m, n)
    initialize_global_variables(m, n)
    for i in xrange(1, m + 1):
        for j in xrange(1, n + 1):
            min_dist = sys.maxint
            min_bptr = 0
            cur_dist = dist[i - 1][j] + 1  # deletion cost.
            if cur_dist < min_dist:
                min_dist = cur_dist
                min_bptr = 1  # 1 means along column-direction.
            cur_dist = dist[i][j - 1] + 1  # insertion cost.
            if cur_dist < min_dist:
                min_dist = cur_dist
                min_bptr = 2  # 2 means alogn row-direction.
            cur_dist = dist[i - 1][j - 1]
            if ref[i - 1] != rec[j - 1]:
                cur_dist = cur_dist + 1  # substitution cost.
            if cur_dist < min_dist:
                min_dist = cur_dist
                min_bptr = 3  # 3 means diagonal-direction
            dist[i][j] = min_dist
            bptr[i][j] = min_bptr


def compute_alignment(ref, rec):
    m = len(ref)
    n = len(rec)
    i = m
    j = n
    res_ref = list()
    res_rec = list()
    while i > 0 or j > 0:
        if bptr[i][j] == 3:
            res_ref.append(ref[i - 1])
            res_rec.append(rec[j - 1])
            i = i - 1
            j = j - 1
        elif bptr[i][j] == 2:
            res_ref.append(None)
            res_rec.append(rec[j - 1])
            j = j - 1
        elif bptr[i][j] == 1:
            res_ref.append(ref[i - 1])
            res_rec.append(None)
            i = i - 1
        else:  # this should not happen.
            log = 'bptr[%d][%d] = %d is unexpected.\n' % \
                  (i, j, bptr[i][j])
            sys.stderr.write(log)
            sys.exit(1)
    res_ref = res_ref[::-1]
    res_rec = res_rec[::-1]
    return res_ref, res_rec


def compute_errors(ref, rec):
    if len(ref) != len(rec):
        log = 'len(ref): %d != len(rec): %d is unexpected.\n' % \
              (len(ref), len(rec))
        sys.stderr.write(log)
        sys.exit(1)

    N = 0  # number of words in reference.
    C = 0  # number of correctness
    S = 0  # number of substitutions
    I = 0  # number of insertions
    D = 0  # number of deletions
    for i in xrange(len(ref)):
        if ref[i] is None:
            I = I + 1
        else:
            N = N + 1
            if rec[i] is None:
                D = D + 1
            elif ref[i] != rec[i]:
                S = S + 1
            else:
                C = C + 1
    return (N, C, S, I, D)


def width(token):
    res = 0
    for char in token:
        if unicodedata.east_asian_width(char) in "AFW":
            res = res + 2
        else:
            res = res + 1
    return res


def space_padding(ref, rec):
    if len(ref) != len(rec):
        log = 'len(ref): %d != len(rec): %d is unexpected.\n' % \
              (len(ref), len(rec))
        sys.stderr.write(log)
        sys.exit(1)

    res_ref = list()
    res_rec = list()
    for i in xrange(len(ref)):
        x = ref[i]
        y = rec[i]
        if x is None:
            x = ' ' * width(y)
        elif y is None:
            y = ' ' * width(x)
        else:
            wx = width(x)
            wy = width(y)
            max_width = max(wx, wy)
            x = x + ' ' * (max_width - wx)
            y = y + ' ' * (max_width - wy)
        res_ref.append(x)
        res_rec.append(y)
    return res_ref, res_rec


def main(args):
    refs = read_text_file(args.reference, args.to_character, args.to_lower)
    N = 0
    C = 0
    S = 0
    I = 0
    D = 0
    for line in sys.stdin:
        line = line.decode('utf-8').strip()
        tokens = parse_text_line(line)
        fid = tokens[0]
        rec = normalize_tokens(tokens[1:], args.to_character, args.to_lower)
        ref = refs[fid]
        compute_distance_backtrace(ref, rec)
        ref, rec = compute_alignment(ref, rec)
        n, c, s, i, d = compute_errors(ref, rec)
        N += n
        C += c
        S += s
        I += i
        D += d
        e = float(n + i - c) / n
        res = '%s wer: %.4f num: %d cor: %d sub: %d ins: %d del: %d' % \
              (fid, e, n, c, s, i, d)
        sys.stdout.write(res + '\n')
        ref, rec = space_padding(ref, rec)
        ref = [fid, 'ref:'] + ref
        res = ' '.join(ref).encode('utf-8')
        sys.stdout.write(res + '\n')
        rec = [fid, 'rec:'] + rec
        res = ' '.join(rec).encode('utf-8')
        sys.stdout.write(res + '\n')

    sys.stdout.write('\n' + '=' * 80 + '\n\n')
    E = float(N + I - C) / N
    res = '%s wer: %.4f num: %d cor: %d sub: %d ins: %d del: %d' % \
          ('summary', E, N, C, S, I, D)
    sys.stdout.write(res + '\n')
    sys.stdout.write('\n' + '=' * 80 + '\n')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    usage = \
        'cat <recognition> | ' + \
        'compute-wer.py ' + \
        '[-h] [--to-character (bool)] [--to-lower (bool)] ' + \
        '<reference>'
    parser = \
        argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            usage=usage,
        )

    # positional args.
    parser.add_argument('reference', metavar = '<reference>',
                        type = str, help = 'reference text.')

    # switches
    parser.add_argument('--to-character', metavar = 'to-character',
                        help='whether to split Chinese words into characters.',
                        type = str2bool, default = 'false')
    parser.add_argument('--to-lower', metavar = 'to-lower',
                        help='whether to convert English words into lower cases',
                        type = str2bool, default = 'false')

    args = parser.parse_args()

    log = ' '.join(sys.argv)
    sys.stderr.write(log + '\n\n')

    main(args)
