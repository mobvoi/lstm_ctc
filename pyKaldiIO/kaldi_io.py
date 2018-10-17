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

import cStringIO
import logging
import os
import subprocess
import sys
from io_funcs import ClassifyRxfilename
from io_funcs import ClassifyWxfilename
from io_funcs import InitKaldiInputStream
from io_funcs import InputType
from io_funcs import LogError
from io_funcs import LogWarning
from io_funcs import OutputType

try:
    from subprocess import DEVNULL  # Python 3.
except ImportError:
    DEVNULL = open(os.devnull, 'wb')

class KaldiInputStream(object):
    """A wrapper of input stream for Kaldi I/O, providing unified interface
    for file/std/pipe inputs and support Peek().
    """
    def __init__(self, stream = None):
        self.stream = None
        self.buffer = None
        if stream:
           self.Open(stream)

    def Open(self, stream):
        if self.stream:
            self.Close()
        self.stream = stream
        self.buffer = cStringIO.StringIO()

    def Close(self):
        if type(self.stream) is subprocess.Popen:
            self.stream.terminate()
        else:
            self.stream.close()
        self.stream = None
        del self.buffer
        self.buffer = None
        return True

    def Peek(self, size = 1):
        oldpos = self.buffer.tell()
        self.buffer.seek(0, os.SEEK_END)
        endpos = self.buffer.tell()
        diff = endpos - oldpos
        if diff < size:
            if type(self.stream) is subprocess.Popen:
                contents = self.stream.stdout.read(size - diff)
            else:
                contents = self.stream.read(size - diff)
            self.buffer.write(contents)
        self.buffer.seek(oldpos)
        res = self.buffer.read(size)
        self.buffer.seek(oldpos)
        return res

    def Read(self, size = None):
        """Read all contents from stream.
        """
        if size is None:
            if type(self.stream) is subprocess.Popen:
                return self.buffer.read() + self.stream.stdout.read()
            else:
                return self.buffer.read() + self.stream.read()
        res = self.buffer.read(size)
        if len(res) < size:
            if type(self.stream) is subprocess.Popen:
                res += self.stream.stdout.read(size - len(res))
            else:
                res += self.stream.read(size - len(res))
        return res

    def Readline(self):
        line = self.buffer.readline()
        if not line.endswith('\n'):
            if type(self.stream) is subprocess.Popen:
                line += self.stream.stdout.readline()
            else:
                line += self.stream.readline()
        return line

    def Seek(self, offset):
        """Seek to the givne offset position for the stream, for files only.

        Args:
            offset: The given offset.

        Returns:
            A boolean variable indicating if the operation is successful.
        """
        if type(self.stream) is not file:
            LogError('stream type \"%s\" does not support seek()'
                     % type(self.stream))
        if self.buffer:
            del self.buffer
        self.buffer = cStringIO.StringIO()
        if self.stream.tell() == offset:
            return True
        self.stream.seek(offset)
        return True

    def Eof(self):
        return not self.Peek(1)


class KaldiOutputStream(object):
    """A wrapper of output stream for Kaldi I/O, providing unified interface
    for file/std/pipe outputs.
    """
    def __init__(self, stream = None):
        self.stream = None
        if stream:
           self.Open(stream)

    def Open(self, stream):
        if self.stream:
            self.Close()
        self.stream = stream

    def Close(self):
        if type(self.stream) is subprocess.Popen:
            self.stream.terminate()
        else:
            self.stream.close()
        self.stream = None
        return True

    def Write(self, data):
        if type(self.stream) is subprocess.Popen:
            self.stream.stdin.write(data)
        else:
            self.stream.write(data)
        return True

    def Flush(self):
        if type(self.stream) is subprocess.Popen:
            self.stream.stdin.flush()
        else:
            self.stream.flush()
        return True


class FileInputImpl(object):
    """Implementation for regular file inputs, such as feats.scp/feats.ark.
    """
    def __init__(self) :
        self.stream = None
        self.filename = None

    def Open(self, rxfilename, binary) :
        """Open a KaldiInputStream for the given rxfilename.

        Args:
            rxfilename: The given filename.
            binary: The opening mode.

        Returns:
            A boolean variable indicating if the operation is successful.
        """
        if self.stream:
            LogError('Called on already open file.')
        self.filename = rxfilename
        mode = 'rt'
        if binary:
            mode = 'rb'
        self.stream = KaldiInputStream(open(rxfilename, mode))
        return True

    def Stream(self):
        if not self.stream:
            LogError('File is not opened')
        return self.stream

    def Close(self):
        if not self.stream:
            LogError('File is not opened')
        return self.stream.Close()

    def MyType(self):
        return InputType.kFileInput


class StandardInputImpl(object):
    """Implementation for standard inputs.
    """
    def __init__(self):
        self.stream = None

    def Open(self, rxfilename, binary):
        """Open a KaldiInputStream for the standard input.

        Args:
            rxfilename: The given filename, should be "" or "-".
            binary: The opening mode, actually does not matter here.

        Returns:
            A boolean variable indicating if the operation is successful.
        """
        if self.stream:
            LogError('Called on already open file.')
        self.stream = KaldiInputStream(sys.stdin)
        return True

    def Stream(self):
        if not self.stream:
            LogError('Object not initialized.')
        return self.stream

    def Close(self):
        if not self.stream:
            LogError('Object not initialized.')
        return self.stream.Close()

    def MyType(self):
        return InputType.kStandardInput


class PipeInputImpl(object):
    """Implementation for pipe inputs, such as 'copy-feats scp:feats.scp ark:-|'.
    """
    def __init__(self) :
        self.stream = None
        self.command = []
        self.process = []

    def Open(self, command, binary, ignore_stderr = True):
        """Open a KaldiInputStream for the given pipe command.

        Args:
            command: The given command for pipe input.
            binary: The opening mode, actually does not matter here.

        Returns:
            A boolean variable indicating if the operation is successful.
        """
        if self.stream:
            LogError('Called on already open file.')
        if not command or command[-1] != '|':
            LogError('Invalid command \"%s\"' % command)
        self.command = command[:-1].split('|')
        stdin = sys.stdin
        stderr = DEVNULL if ignore_stderr else sys.stderr
        for i in xrange(len(self.command)):
            self.process.append(subprocess.Popen(self.command[i].split(),
                                                 stdin=stdin,
                                                 stdout=subprocess.PIPE,
                                                 stderr=stderr))
            stdin = self.process[-1].stdout
        self.stream = KaldiInputStream(self.process[-1])
        return True

    def Stream(self):
        if not self.stream:
            LogError('Object not initialized.')
        return self.stream

    def Close(self):
        if not self.stream:
            LogError('Object not initialized.')
        return self.stream.Close()

    def MyType(self):
        return InputType.kPipeInput


class OffsetFileInputImpl(object):
    """Implementation for file inputs with offset, such as file.ark:123.
    """
    def __init__(self) :
        self.stream = None
        self.filename = None
        self.offset = None
        self.binary = None

    def Open(self, rxfilename, binary):
        """Open a KaldiInputStream for the given filename and offset.

        Args:
            rxfilename: The given filename including offset.
            binary: The opening mode.

        Returns:
            A boolean variable indicating if the operation is successful.
        """
        (filename, offset) = self.SplitFilename(rxfilename)
        if self.stream:
            if self.filename == filename and self.binary == binary:
                if not self.Seek(offset):
                    LogError('Invalid offset = \"%d\"' % offset)
                return True
            else:
                self.Close()
        self.filename = filename
        self.binary = binary
        mode = 'rt'
        if binary:
            mode = 'rb'
        self.stream = KaldiInputStream(open(self.filename, mode))
        if not self.Seek(offset):
            LogError('Invalid offset = \"%d\"' % offset)
        return True

    def Stream(self):
        if not self.stream:
            LogError('File is not opened.')
        return self.stream

    def Close(self):
        if not self.stream:
            LogError('File is not opened.')
        return self.stream.Close()

    def MyType(self):
        return InputType.kOffsetFileInput

    def SplitFilename(self, rxfilename):
        pos = rxfilename.rfind(':')
        if pos < 0:
            LogError('Invalid rxfilename \"%s\"' % rxfilename)
        filename = rxfilename[:pos]
        offset = int(rxfilename[pos+1:])
        return (filename, offset)

    def Seek(self, offset):
        if not self.stream.Seek(offset):
            LogError('Invalid offset = \"%d\"' % offset)
        self.offset = offset
        return True


class Input(object):
    """A wrapper of differnt input streams for Kaldi I/O.
    """
    def __init__(self, rxfilename = None):
        self.impl = None
        self.binary = None
        # Explicitly compare with None since rxfilename == '' is still a valid
        # input (kStandardInput)
        if rxfilename is not None:
            if not self.Open(rxfilename):
                LogError('Failed opening input stream')

    def Open(self, rxfilename):
        return self.OpenInternal(rxfilename, True)

    def IsOpen(self):
        return self.impl != None

    def IsBinary(self):
        return self.binary

    def Close(self):
        if self.impl != None:
            ans = self.impl.Close()
            del self.impl
            self.impl = None
            self.binary = None
            return ans
        else:
            return True

    def Stream(self):
        if not self.impl:
            LogError('No input specified.')
        return self.impl.Stream()

    def OpenInternal(self, rxfilename, file_binary):
        """Open a KaldiInputStream for the given rxfilename.

        Args:
            rxfilename: The given filename.
            binary: The opening mode.

        Returns:
            A boolean variable indicating if the operation is successful.
        """
        rxtype = ClassifyRxfilename(rxfilename)
        if self.IsOpen():  # May have to close the stream first.
            if rxtype == InputType.kOffsetFileInput and \
               self.impl.MyType() == InputType.kOffsetFileInput:
                # We want to use the same object to Open(), this is in case
                # the files are the same, so we can just seek.
                if not self.impl.Open(rxfilename, file_binary):
                    del self.impl
                    self.impl = None
                    return False
                # set self.binary.
                self.binary = InitKaldiInputStream(self.impl.Stream())
                return True
            else:
                self.Close()
        if rxtype == InputType.kFileInput:
            self.impl = FileInputImpl()
        elif rxtype == InputType.kStandardInput:
            self.impl = StandardInputImpl()
        elif rxtype == InputType.kPipeInput:
            self.impl = PipeInputImpl()
        elif rxtype == InputType.kOffsetFileInput:
            self.impl = OffsetFileInputImpl()
        elif rxtype == InputType.kNoInput:
            LogError('Invalid input filename format \"%s\".' % rxfilename)
        else:
            LogError('Unknown input type.')
        if not self.impl.Open(rxfilename, self.binary):
            self.Close()
            return False
        else:
            self.binary = InitKaldiInputStream(self.impl.Stream())
            return True


class FileOutputImpl(object):
    """Implementation for regular file outputs, such as feats.ark.
    """
    def __init__(self) :
        self.stream = None
        self.filename = None

    def Open(self, wxfilename, binary) :
        """Open a KaldiOutputStream for the given wxfilename.

        Args:
            wxfilename: The given filename.
            binary: The opening mode.

        Returns:
            A boolean variable indicating if the operation is successful.
        """
        if self.stream:
            LogError('Called on already open file.')
        self.filename = wxfilename
        mode = 'w'
        if binary:
            mode = 'wb'
        self.stream = KaldiOutputStream(open(wxfilename, mode))
        return True

    def Stream(self):
        if not self.stream:
            LogError('File is not opened')
        return self.stream

    def Close(self):
        if not self.stream:
            LogError('File is not opened')
        return self.stream.Close()

    def MyType(self):
        return OutputType.kFileOutput


class StandardOutputImpl(object):
    """Implementation for standard outputs.
    """
    def __init__(self):
        self.stream = None

    def Open(self, wxfilename, binary):
        """Open a KaldiOutputStream for the standard output.

        Args:
            wxfilename: The given filename, should be "" or "-".
            binary: The opening mode, actually does not matter here.

        Returns:
            A boolean variable indicating if the operation is successful.
        """
        if self.stream:
            LogError('Called on already open file.')
        self.stream = KaldiOutputStream(sys.stdout)
        return True

    def Stream(self):
        if not self.stream:
            LogError('Object not initialized.')
        return self.stream

    def Close(self):
        if not self.stream:
            LogError('Object not initialized.')
        self.stream.Flush()
        return self.stream.Close()

    def MyType(self):
        return OutputType.kStandardOutput


class PipeOutputImpl(object):
    """Implementation for pipe outputs, such as '| gzip -c > feats.gz'.
    """
    def __init__(self) :
        self.stream = None
        self.command = []
        self.process = []

    def Open(self, command, binary, ignore_stderr = True):
        """Open a KaldiOutputStream for the given pipe command.

        Args:
            command: The given command for pipe output.
            binary: The opening mode, actually does not matter here.

        Returns:
            A boolean variable indicating if the operation is successful.
        """
        if self.stream:
            LogError('Called on already open file.')
        if not command or command[0] != '|':
            LogError('Invalid command \"%s\"' % command)
        self.command = command[1:].split('|')
        # TODO(cfyeh): make sure the pipes are concatenated correctly
        stdin = sys.stdin
        stderr = DEVNULL if ignore_stderr else sys.stderr
        for i in xrange(len(self.command)):
            self.process.append(subprocess.Popen(self.command[i].split(),
                                                 stdin=stdin,
                                                 stdout=subprocess.PIPE,
                                                 stderr=stderr))
            stdin = self.process[-1].stdout
        self.stream = KaldiOutputStream(self.process[0])
        return True

    def Stream(self):
        if not self.stream:
            LogError('Object not initialized.')
        return self.stream

    def Close(self):
        if not self.stream:
            LogError('Object not initialized.')
        self.stream.Flush()
        return self.stream.Close()

    def MyType(self):
        return OutputType.kPipeOutput


class Output(object):
    """A wrapper of differnt output streams for Kaldi I/O.
    """
    def __init__(self, wxfilename = None, binary = None, write_header = False):
        self.impl = None
        self.binary = None
        # Explicitly compare with None since wxfilename == '' is still a valid
        # input (kStandardInput)
        if wxfilename is not None and binary is not None:
            if not self.Open(wxfilename, binary, write_header):
                LogError('Failed opening output stream')

    def Open(self, wxfilename, binary, header = False):
        if self.IsOpen():
            if not self.Close():
                LogError('failed to close output stream')
        wxtype = ClassifyWxfilename(wxfilename)
        if wxtype == OutputType.kFileOutput:
            self.impl = FileOutputImpl()
        elif wxtype == OutputType.kStandardOutput:
            self.impl = StandardOutputImpl()
        elif wxtype == OutputType.kPipeOutput:
            self.impl = PipeOutputImpl()
        else:
            LogWarning('Invalid output filename \"%s\"' % wxfilename)
            return False
        if not self.impl.Open(wxfilename, binary):
            del self.impl
            self.impl = None
            return False
        else:
            if header:
                InitKaldiOutputStream(self.impl.Stream(), binary)
            return True

    def IsOpen(self):
        return self.impl != None

    def Close(self):
        if self.impl != None:
            ans = self.impl.Close()
            del self.impl
            self.impl = None
            self.binary = None
            return ans
        else:
            return False

    def Stream(self):
        if not self.impl:
            LogError('No output specified.')
        return self.impl.Stream()
