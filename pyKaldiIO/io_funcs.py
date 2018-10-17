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

import inspect
import logging
import os
import struct
import sys
from enum import Enum


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)-15s %(name)-5s %(levelname)-8s: %(message)s')


def GetLoggerPrefix(stack):
    file_name = os.path.basename(stack[1][0].f_code.co_filename)
    line_number = stack[1][0].f_lineno
    func_name = stack[1][0].f_code.co_name
    if 'self' in stack[1][0].f_locals:
        class_name = inspect.stack()[2][0].f_locals["self"].__class__.__name__
        res = '%s:%s %s.%s()' %(file_name, line_number, class_name, func_name)
    else:
        res = '%s:%s %s()' %(file_name, line_number, func_name)
    return res


def LogError(msg):
    logger = logging.getLogger(GetLoggerPrefix(inspect.stack()))
    logger.error(msg)
    sys.exit(1)


def LogWarning(msg):
    logger = logging.getLogger(GetLoggerPrefix(inspect.stack()))
    logger.warning(msg)


def LogInfo(msg):
    logger = logging.getLogger(GetLoggerPrefix(inspect.stack()))
    logger.info(msg)


def LogDebug(msg):
    logger = logging.getLogger(GetLoggerPrefix(inspect.stack()))
    logger.debug(msg)


def InitKaldiInputStream(stream):
    """Determine if an opened stream is good for reading and is binary or not.

    Args:
        stream: An opened KaldiInputStream.

    Returns:
        A boolean variable indicating if the input stream is in binary format.
    """
    c = stream.Peek(1)
    if not c:
        LogError('Error reading from stream (end of stream).')
    if c == '\0':  # seems to be binary
        stream.Read(1)
        c = stream.Peek(1)
        if not c:
            LogError('Error reading from stream (end of stream).')
        if c != 'B':
            LogError('Cannot determine if the input stream is in binary or not.')
        stream.Read(1)
        return True
    else:
        return False


def InitKaldiOutputStream(stream, binary):
    """Initialize an opened stream for writing by writing an optional binary
    header and modifying the floating-point precision.

    Args:
        stream: An opened KaldiOutputStream.
        binary: Whether to put a binary indicator in the output stream.

    Returns:
        A boolean variable indicating if the input stream is in binary format.
    """
    if binary:
        stream.Write('\0')
        stream.Write('B')
    return True


def ReadToken(stream, binary, remove_tail_space = True):
    """Read string token from input stream.

    Args:
        stream: An opened KaldiInputStream.
        binary: If the input stream is in binary.
        remove_tail_space: Whether to remove the tailing space.

    Returns:
        A string read from the input stream.
    """
    res = []
    if not binary :
        while True:
            c = stream.Peek(1)
            if not c:
                return ''.join(res)
            if c == ' ' or c == '\n':
                stream.Read(1)
            else:
                break
    while True:
        c = stream.Peek(1)
        if not c:
            return ''.join(res)
        if c == ' ' or c == '\n':
            break
        else:
            res.append(c)
            stream.Read(1)
    if remove_tail_space:
        if stream.Peek(1) == ' ':
            stream.Read(1)
    return ''.join(res)


def ExpectToken(stream, binary, token):
    if not binary and stream.Peek(1) == ' ':
        stream.Read(1)
    r = ReadToken(stream, binary, False)
    if stream.Peek(1) == ' ':
        stream.Read(1)
    if r != token:
        LogError('Expected token \"%s\", got instead \"%s\"' % (token, r))


def WriteToken(stream, binary, data):
    """Write token to output stream.

    Args:
        stream: An opened KaldiOutputStream.
        binary: Whether to write in binary. (ignored here)
        data: The string to write.

    Returns:
        A boolean variable indicating if the input stream is in binary format.
    """
    if not data:
        LogError('Invalid data to write \"%s\"' % data)
    stream.Write('%s ' % data)
    return True


class BasicType(Enum):
    """Enumerations for basic types in C/C++.
    """
    notype = 0
    cchar = 1
    cint8 = 2
    cint16 = 3
    cint32 = 4
    cint64 = 5
    cuint8 = 6
    cuint16 = 7
    cuint32 = 8
    cuint64 = 9
    cfloat = 10
    cdouble = 11


def ReadBasicType(stream, binary, ctype):
    """Read basic type from input stream.

    Args:
        stream: An opened KaldiInputStream.
        binary: If the input stream is in binary.
        ctype: The basic type to read (BasicType.{cint32/cfloat/...}).

    Returns:
        An basic type variable read from the input stream.
    """
    if ctype == BasicType.notype:
        LogError('Invalid basic type \"%s\"' % ctype)
    if binary:
        len_c_in = stream.Read(1)
        # TODO(cfyeh): implement other basic types.
        if ctype == BasicType.cint32:
            res = struct.unpack('i', stream.Read(4))[0]
        elif ctype == BasicType.cuint8:
            res = struct.unpack('B', stream.Read(1))[0]
        elif ctype == BasicType.cuint16:
            res = struct.unpack('H', stream.Read(2))[0]
        elif ctype == BasicType.cfloat:
            res = struct.unpack('f', stream.Read(4))[0]
        else:
            LogError('Type \"%s\" not implemented yet.' % ctype)
        return res
    else:
        # TODO(cfyeh): implement text mode.
        LogError('Text mode not implemented yet.')


def ReadInt32(stream, binary):
    return ReadBasicType(stream, binary, BasicType.cint32)


def ReadUint8(stream, binary):
    return ReadBasicType(stream, binary, BasicType.cuint8)


def ReadUint32(stream, binary):
    return ReadBasicType(stream, binary, BasicType.cuint16)


def ReadFloat(stream, binary):
    return ReadBasicType(stream, binary, BasicType.cfloat)


def WriteBasicType(stream, binary, ctype, value):
    """Write basic type to output stream.

    Args:
        stream: An opened KaldiOutputStream.
        binary: Whether to write in binary.
        ctype: The basic type to write (BasicType.{cint32/cfloat/...}).
        value: The value to write

    Returns:
        An basic type variable read from the input stream.
    """
    if ctype == BasicType.notype:
        LogError('Invalid basic type \"%s\"' % ctype)
    if binary:
        # TODO(cfyeh): implement other basic types.
        if ctype == BasicType.cint32:
            stream.Write(struct.pack('c', chr(4)))
            stream.Write(struct.pack('i', value))
        else:
            LogError('Type \"%s\" not implemented yet.' % ctype)
    else:
        stream.Write('%s ' % value)


class InputType(Enum):
    """Enumerations for input types in Kaldi.
    """
    kNoInput = 0  # Invalid filenames (leading or trailing space, things that
                  # look like wspecifiers and rspecifiers or pipes to write to
                  # with trailing |.
    kFileInput = 1  # Normal filenames.
    kStandardInput = 2  # The empty string or "-".
    kPipeInput = 3  # e.g. "gzip -c blah.gz |".
    kOffsetFileInput = 4  # Offsets into files, e.g. /some/filename:12970.

def ClassifyRxfilename(filename):
    """Interprets the input type for the given filename.

    Args:
        filename: A string indicating the filename.

    Returns:
        InputType for the given filename.
    """
    if not filename or filename == '-':
        return InputType.kStandardInput
    if filename.startswith('|'):
        return InputType.kNoInput
    if filename.startswith(' ') or filename.endswith(' '):
        return InputType.kNoInput
    if filename.startswith('t,') or filename.startswith('b,'):
        return InputType.kNoInput
    if filename.endswith('|'):
        return InputType.kPipeInput
    # OK, it could be an offset into a file
    if filename[-1].isdigit():
        i = -1
        while i + len(filename) > 0 and filename[i].isdigit():
            i = i - 1
        # Filename is like some_file:12345
        if filename[i] == ':':
            return InputType.kOffsetFileInput
        else:
            return InputType.kFileInput
    else:
        # At this point it matched no other pattern so we assume a filename,
        # but we check for '|' as it's a common source of errors to have pipe
        # commands without the pipe in the right place. Say that it can't be
        # classified in this case.
        if '|' in filename :
            LogError('Trying to classify rxfilename with pipe symbol in the '
                     'wrong place (pipe without | at the end?): \"%s\"'
                     % filename)
        # kFileInput: Matched no other pattern assume an actual filename.
        return InputType.kFileInput


class RspecifierOptions(object):
    """Options for Kaldi rspecifier.
    """
    def __init__(self):
        # These options only make a difference for the RandomAccessTableReader
        # class.
        self.once = False  # We assert that the program will only ask for each
                           # key once.
        self.sorted = False  # We assert that the keys are sorted.
        self.called_sorted = False  # We assert that the HasKey(), Value()
                                    # functions will also be called in sorted
                                    # order.
                                    # [this implies "once" but not vice versa].
        self.permissive = False  # If "permissive", when reading from scp files
                                 # it treats scp files that can't be read as if
                                 # the corresponding key were not there.
                                 # For archive files it will suppress errors
                                 # getting thrown if the archive is corrupted
                                 # and can't be read to the end.
        self.background = False  # For sequential readers, if the background
                                 # option ("bg") is provided, it will read ahead
                                 # to the next object in a background thread.


class RspecifierType(Enum):
    """Enumerations for rspecifier types in Kaldi.
    """
    kNoRspecifier = 0
    kArchiveRspecifier = 1
    kScriptRspecifier = 2


def ClassifyRspecifier(rspecifier):
    """Interprets type / filename / options for the given rspecifier.

    Args:
        rspecifier: A string indicating the rspecifier.

    Returns:
        (RspecifierType, filename, RspecifierOptions) for the given filename.

    Note:
        We also allow the meaningless prefixes b, and t, plus the options
        o (once), no (not-once), s (sorted) and ns (not-sorted), p (permissive)
        and np (not-permissive). So the following would be valid:

        f, o, b, np, ark:rxfilename  ->  RspecifierType.kArchiveRspecifier

    Examples:
        b, ark:rxfilename -> RspecifierType.kArchiveRspecifier
        t, ark:rxfilename -> RspecifierType.kArchiveRspecifier
        b, scp:rxfilename -> RspecifierType.kScriptRspecifier
        t, no, s, scp:rxfilename -> RspecifierType.kScriptRspecifier
        t, ns, scp:rxfilename -> RspecifierType.kScriptRspecifier

    Improperly formed Rspecifiers will be classified as RspecifierType.kNoRspecifier.
    """
    rxfilename = ''
    opts = RspecifierOptions()
    rspecifier_type = RspecifierType.kNoRspecifier
    pos = rspecifier.find(':')
    if pos < 0 or rspecifier[-1] == ' ':
        return (rspecifier_type, rxfilename, opts)
    before_colon = rspecifier[0:pos]
    after_colon = rspecifier[pos+1:]
    for part in before_colon.replace(',', ' ').split():
        if part == 'b' or part == 't':
            continue
        elif part == 'o':
            opts.once = True
        elif part == 'no':
            opts.once = False
        elif part == 'p':
            opts.permissive = True
        elif part == 'np':
            opts.permissive = False
        elif part == 's':
            opts.sorted = True
        elif part == 'ns':
            opts.sorted = False
        elif part == 'cs':
            opts.called_sorted = True
        elif part == 'ncs':
            opts.called_sorted = False
        elif part == 'bg':
            opts.background = True
        elif part == 'ark':
            if rspecifier_type == RspecifierType.kNoRspecifier:
                rspecifier_type = RspecifierType.kArchiveRspecifier
            else:
                # Repeated or combined ark and scp options are invalid.
                rspecifier_type = RspecifierType.kNoRspecifier
                return (rspecifier_type, rxfilename, opts)
        elif part == 'scp':
            if rspecifier_type == RspecifierType.kNoRspecifier:
                rspecifier_type = RspecifierType.kScriptRspecifier
            else:
                # Repeated or combined ark and scp options are invalid.
                rspecifier_type = RspecifierType.kNoRspecifier
                return (rspecifier_type, rxfilename, opts)
        else:
            # Could not interpret this option.
            return (rspecifier_type, rxfilename, opts)
    if rspecifier_type == RspecifierType.kArchiveRspecifier or \
       rspecifier_type == RspecifierType.kScriptRspecifier:
        rxfilename = after_colon
    return (rspecifier_type, rxfilename, opts)


class OutputType(Enum):
    """Enumerations for output types in Kaldi.
    """
    kNoOutput = 0  # Invalid filenames (leading or trailing space, things that
                   # look like wspecifiers and rspecifiers or pipes to write to
                   # with trailing |.
    kFileOutput = 1  # Normal filenames.
    kStandardOutput = 2  # The empty string or "-".
    kPipeOutput = 3  # e.g. "| gzip -c > blah.gz".


def ClassifyWxfilename(filename):
    """Interprets the output type for the given filename.

    Args:
        filename: A string indicating the filename.

    Returns:
        OutputType for the given filename.
    """
    if not filename or filename == '-':
        return OutputType.kStandardOutput
    if filename.startswith('|'):
        return OutputType.kPipeOutput
    if filename.startswith(' ') or filename.endswith(' '):
        return OutputType.kNoOutput
    if filename.startswith('t,') or filename.startswith('b,'):
        return OutputType.kNoOutput
    if filename.endswith('|'):
        return OutputType.kNoOutput
    if filename[-1].isdigit():
        i = -1
        while i + len(filename) > 0 and filename[i].isdigit():
            i = i - 1
        # Filename is like some_file:12345, not allowed.
        if filename[i] == ':':
            return OutputType.kNoOutput
        else:
            return OutputType.kFileOutput
    else:
        # At this point it matched no other pattern so we assume a filename,
        # but we check for '|' as it's a common source of errors to have pipe
        # commands without the pipe in the right place. Say that it can't be
        # classified in this case.
        if '|' in filename :
            LogError('Trying to classify wxfilename with pipe symbol in the '
                     'wrong place (pipe without | at the beginning?): \"%s\"'
                     % filename)
        # kFileInput: Matched no other pattern assume an actual filename.
        return OutputType.kFileOutput


class WspecifierOptions(object):
    """Options for Kaldi wspecifier.
    """
    def __init__(self):
        self.binary = True
        self.flush = False
        self.permissive = False  # Will ignore absent scp entries.


class WspecifierType(Enum):
    """Enumerations for wspecifier types in Kaldi.
    """
    kNoWspecifier = 0
    kArchiveWspecifier = 1
    kScriptWspecifier = 2
    kBothWspecifier = 3


def ClassifyWspecifier(wspecifier):
    """Interprets type / filenames / options for the given wspecifier.

    Args:
        wspecifier: A string indicating the wspecifier.

    Returns:
        (WspecifierType, archive_filename, script_filename, WspecifierOptions)
        for the given filename.

    Examples:
        ark,t:wxfilename -> kArchiveWspecifier
        ark,b:wxfilename -> kArchiveWspecifier
        scp,t:rxfilename -> kScriptWspecifier
        scp,t:rxfilename -> kScriptWspecifier
        ark,scp,t:filename, wxfilename -> kBothWspecifier
        ark,scp:filename, wxfilename ->  kBothWspecifier

        Note we can include the flush option (f) or no-flush (nf) anywhere: e.g.
        ark,scp,f:filename, wxfilename ->  kBothWspecifier or:
        scp,t,nf:rxfilename -> kScriptWspecifier

    Improperly formed Wspecifiers will be classified as WspecifierType.kNoWspecifier.
    """
    archive_filename = ''
    script_filename = ''
    opts = WspecifierOptions()
    wspecifier_type = WspecifierType.kNoWspecifier
    pos = wspecifier.find(':')
    if pos < 0 or wspecifier[-1] == ' ':
        return (wspecifier_type, archive_filename, script_filename, opts)
    before_colon = wspecifier[0:pos]
    after_colon = wspecifier[pos+1:]
    for part in before_colon.replace(',', ' ').split():
        if part == 'b':
            opts.binary = True
        elif part == 't':
            opts.binary = False
        elif part == 'f':
            opts.flush = True
        elif part == 'nf':
            opts.flush = False
        elif part == 'p':
            opts.permissive = True
        elif part == 'ark':
            if wspecifier_type == WspecifierType.kNoWspecifier:
                wspecifier_type = WspecifierType.kArchiveWspecifier
            else:
                # Repeated or combined ark and scp options are invalid.
                wspecifier_type = WspecifierType.kNoWspecifier
                return (Wspecifier_type, archive_filename, script_filename, opts)
        elif part == 'scp':
            if wspecifier_type == WspecifierType.kNoWspecifier:
                wspecifier_type = WspecifierType.kScriptWspecifier
            elif wspecifier_type == WspecifierType.kArchiveWspecifier:
                wspecifier_type = WspecifierType.kBothWspecifier
            else:
                # Repeated or combined ark and scp options are invalid.
                wspecifier_type = WspecifierType.kNoWspecifier
                return (wspecifier_type, archive_filename, script_filename, opts)
        else:
            # Could not interpret this option.
            return (wspecifier_type, archive_filename, script_filename, opts)
    if wspecifier_type == WspecifierType.kArchiveWspecifier:
        archive_filename = after_colon
    elif wspecifier_type == WspecifierType.kScriptWspecifier:
        script_filename = after_colon
    elif wspecifier_type == WspecifierType.kBothWspecifier:
        pos = after_colon.find(',')
        if pos < 0:
            return (wspecifier_type, archive_filename, script_filename, opts)
        archive_filename = after_colon[0:pos]
        script_filename = after_colon[pos+1:]
    else:
        pass
    return (wspecifier_type, archive_filename, script_filename, opts)
