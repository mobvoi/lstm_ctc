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

import bisect
import logging
import kaldi_holder
import kaldi_io
from enum import Enum
from io_funcs import ClassifyRspecifier
from io_funcs import ClassifyWspecifier
from io_funcs import InitKaldiInputStream
from io_funcs import LogError
from io_funcs import LogWarning
from io_funcs import ReadToken
from io_funcs import RspecifierType
from io_funcs import WspecifierType
from kaldi_holder import HolderType
from kaldi_holder import FloatMatrixHolder
from kaldi_holder import NewHolderByType
from kaldi_holder import PosteriorHolder
from kaldi_holder import WriteHolderValueToStream
from kaldi_io import Input
from kaldi_io import Output
from operator import itemgetter
from text_util import IsToken


class SequentialTableReaderStateType(Enum):
    """Enumerations for SequentialTableReader state types.
    """
    kUninitialized = 0  # Uninitialized or closed.
    kFileStart = 1  # [state we use internally: just opened.]
    kEof = 2  # We did Next() and found eof in archive
    kError = 3  # Some other error
    kHaveScpLine = 4  # Have a line of the script file but nothing else.
    kHaveObject = 5  # We read the key and the object after it
    kHaveRange = 6  # we have the range object in range_holder_ (implies range_
                    # nonempty).
    kFreedObject = 7  # The user called FreeCurrent().


class SequentialTableReaderArchiveImpl(object):
    def __init__(self, holder_type):
        """Initialize the reader for the given holder type.

        Args:
            holder_type: The given holder type.
        """
        self.rspecifier = None
        self.opts = None
        self.archive_rxfilename = None
        self.input = Input()
        self.type = holder_type
        self.holder = NewHolderByType(self.type)
        self.key = None
        self.state = SequentialTableReaderStateType.kUninitialized

    def Open(self, rspecifier):
        """Open a reader for the given rspecifier.

        Args:
            rspecifier: The given rspecifier.

        Returns:
            A boolean variable indicating if the operation is successful.
        """
        if self.state != SequentialTableReaderStateType.kUninitialized:
            # call Close() yourself to suppress this exception.
            if not self.Close():
                if self.opts.permissive:
                    LogWarning('Error closing previous input (only warning, '
                               'since permissive mode).')
                else:
                    LogError('Error closing previous input, rspecifier was '
                             '\"%s\"' % self.rspecifier)
        self.rspecifier = rspecifier
        (rspecifier_type, rxfilename, opts) = ClassifyRspecifier(rspecifier)
        self.archive_rxfilename = rxfilename
        self.opts = opts
        if rspecifier_type != RspecifierType.kArchiveRspecifier:
            LogError('Invalid rspecifier type \"%s\"' % rspecifier_type)
        self.input = Input()
        if self.holder.IsReadInBinary():
            success = self.input.Open(self.archive_rxfilename)
        else:
            success = self.input.OpenTextMode(self.archive_rxfilename)
        if not success:
            self.state = SequentialTableReaderStateType.kUninitialized
            LogError('Failed to open stream \"%s\"' % self.archive_rxfilename)
        self.state = SequentialTableReaderStateType.kFileStart
        self.Next()
        if self.state == SequentialTableReaderStateType.kError:
            self.input.Close()
            self.state = SequentialTableReaderStateType.kUninitialized
            LogError('Error beginning to read archive file \"%s\" (wrong '
                     'filename?)' % self.archive_rxfilename)
        if self.state != SequentialTableReaderStateType.kHaveObject and \
           self.state != SequentialTableReaderStateType.kEof:
            LogError('Invalid state \"%s\"' % self.state)
        return True

    def Next(self):
        if self.state == SequentialTableReaderStateType.kHaveObject:
            self.holder.Clear()
        elif self.state == SequentialTableReaderStateType.kFileStart or \
             self.state == SequentialTableReaderStateType.kFreedObject:
            pass
        else:
            LogError('Invalid state \"%s\"' % self.state)
        if self.input.Stream().Eof():
            self.state = SequentialTableReaderStateType.kEof
            return True
        self.key = ReadToken(self.input.Stream(),
                             self.input.IsBinary(),
                             False)
        c = self.input.Stream().Peek(1)
        # We expect a space ' ' after the key. We also allow tab, just so we
        # can read archives generated by scripts that may not be fully aware
        # of how this format works.
        if c != ' ' and c != '\t' and c != '\n':
            LogError('Invalid archive file format: expected space after key '
                     '\"%s\", got character \"%s\" when reading archive '
                     '\"%s\".' % (self.key, c, self.archive_rxfilename))
        if c != '\n':  # Consume the space or tab.
            self.input.Stream().Read(1)
        binary = InitKaldiInputStream(self.input.Stream())
        if not self.holder.Read(self.input.Stream(), binary):
            self.holder.Clear()
            LogError('Failed to read object from archive \"%s\"'
                     % self.archive_rxfilename)
        self.state = SequentialTableReaderStateType.kHaveObject
        return True

    def IsOpen(self):
        if self.state == SequentialTableReaderStateType.kEof or \
           self.state == SequentialTableReaderStateType.kHaveObject or \
           self.state == SequentialTableReaderStateType.kFreedObject:
            return True
        elif self.state == SequentialTableReaderStateType.kUninitialized:
            return False
        else:
            # note: kFileStart is not a valid state for the user to call a
            # member function (we never return from a public function in
            # this state).
            LogError('Invalid state \"%s\"' % self.state)

    def Done(self):
        if self.state == SequentialTableReaderStateType.kHaveObject:
            return False
        elif self.state == SequentialTableReaderStateType.kEof or \
             self.state == SequentialTableReaderStateType.kError:
            # Error condition, like Eof, counts as Done(); the
            # destructor/Close() will inform the user of the error.
            return True
        else:
            LogError('Invalid state \"%s\"' % self.state)

    def Key(self):
        if self.state != SequentialTableReaderStateType.kHaveObject:
            LogError('Invalid state \"%s\"' % self.state)
        return self.key

    def Value(self):
        if self.state != SequentialTableReaderStateType.kHaveObject:
            LogError('Invalid state \"%s\"' % self.state)
        return self.holder.Value()

    def Close(self):
        if not self.IsOpen():
            LogError('Called on input that was not open.')
        status = 0
        if self.input.IsOpen():
            status = self.input.Close()
        if self.state == SequentialTableReaderStateType.kHaveObject:
            self.holder.Clear()
        old_state = self.state
        self.state = SequentialTableReaderStateType.kUninitialized
        if old_state == SequentialTableReaderStateType.kError or \
            (old_state == SequentialTableReaderStateType.kEof and
             status != 0):
            if self.opts.permissive:
                LogWarning('Error state detected closing reader. Ignoring '
                           'it because you specified permissive mode.')
                return True
            else:
                return False
        else:
            return True


class SequentialTableReaderScriptImpl(object):
    def __init__(self, holder_type):
        """Initialize the reader for the given holder type.

        Args:
            holder_type: The given holder type.
        """
        self.rspecifier = None
        self.opts = None
        self.script_rxfilename = None
        self.script_input = Input()
        self.data_input = Input()
        self.type = holder_type
        self.holder = NewHolderByType(self.type)
        self.range_holder = NewHolderByType(self.type)
        self.key = None
        self.data_rxfilename = None
        self.range = None
        self.state = SequentialTableReaderStateType.kUninitialized

    def Open(self, rspecifier):
        """Open a reader for the given rspecifier.

        Args:
            rspecifier: The given rspecifier.

        Returns:
            A boolean variable indicating if the operation is successful.
        """
        # You may call Open from states kUninitialized and kError.
        # It may leave the object in any of the states.
        if self.state != SequentialTableReaderStateType.kUninitialized and \
           self.state != SequentialTableReaderStateType.kError:
            # call Close() yourself to suppress this exception.
            if not self.Close():
                LogError('Error closing previous input, rspecifier was \"%s\"'
                         % self.rspecifier)
        self.rspecifier = rspecifier
        (rspecifier_type, rxfilename, opts) = ClassifyRspecifier(rspecifier)
        self.script_rxfilename = rxfilename
        self.opts = opts
        if rspecifier_type != RspecifierType.kScriptRspecifier:
            LogError('Invalid rspecifier type \"%s\"' % rspecifier_type)
        self.script_input = Input()
        if not self.script_input.Open(self.script_rxfilename):
            LogError('Failed opening script file \"%s\"'
                     % self.script_rxfilename)
        if self.script_input.IsBinary():
            self.SetErrorState()
            LogError('script file should not be in binary format.')
        else:
            self.state = SequentialTableReaderStateType.kFileStart
            self.Next()
            if self.state == SequentialTableReaderStateType.kError:
                return False
            # any other status, including kEof, is OK from the point of view
            # of the 'open' function (empty scp file is not inherently an
            # error).
            return True

    def IsOpen(self):
        if self.state == SequentialTableReaderStateType.kEof or \
           self.state == SequentialTableReaderStateType.kHaveScpLine or \
           self.state == SequentialTableReaderStateType.kHaveObject or \
           self.state == SequentialTableReaderStateType.kHaveRange:
            return True
        elif self.state == SequentialTableReaderStateType.kUninitialized or \
             self.state == SequentialTableReaderStateType.kError:
            return False
        else:
            # note: kFileStart is not a valid state for the user to call a
            # member function (we never return from a public function in
            # this state).
            LogError('Invalid state \"%s\"' % self.state)

    def Done(self):
        if self.state == SequentialTableReaderStateType.kHaveScpLine or \
           self.state == SequentialTableReaderStateType.kHaveObject or \
           self.state == SequentialTableReaderStateType.kHaveRange:
            return False
        elif self.state == SequentialTableReaderStateType.kEof or \
             self.state == SequentialTableReaderStateType.kError:
            # Error condition, like Eof, counts as Done(); the
            # destructor/Close() will inform the user of the error.
            return True
        else:
            LogError('Invalid state \"%s\"' % self.state)

    def Key(self):
        if self.state != SequentialTableReaderStateType.kHaveScpLine and \
           self.state != SequentialTableReaderStateType.kHaveObject and \
           self.state != SequentialTableReaderStateType.kHaveRange:
            LogError('Invalid state \"%s\"' % self.state)
        return self.key

    def Value(self):
        if not self.EnsureObjectLoaded():
            LogError('Failed to load object from \"%s\" to suppress this '
                     'error, add the permissive (p, ) option to the '
                     'rspecifier.' % self.data_rxfilename)
        if self.state == SequentialTableReaderStateType.kHaveRange:
            return self.range_holder.Value()
        elif self.state == SequentialTableReaderStateType.kHaveObject:
            return self.holder.Value()
        else:
            LogError('Invalid state \"%s\"' % self.state)

    def Next(self):
        while True:
            self.NextScpLine()
            if self.Done():
                return
            if self.opts.permissive:
                # Permissive mode means, when reading scp files, we treat keys
                # whose scp entry cannot be read as nonexistent.  This means
                # trying to read.
                if self.EnsureObjectLoaded():
                    return # Success.
                # else try the next scp line.
            else:
                # We go the next key; Value() will crash if we can't read the
                # object on the scp line.
                return

    def Close(self):
        status = 0
        if self.script_input.IsOpen():
            status = self.script_input.Close()
        if self.data_input.IsOpen():
            self.data_input.Close()
        self.range_holder.Clear()
        self.holder.Clear()
        if not self.IsOpen():
            LogError('Called on input that was not open.')

    def SetErrorState(self):
        self.state = SequentialTableReaderStateType.kError
        self.script_input.Close()
        self.data_input.Close()
        self.holder.Clear()
        self.range_holder.Clear()
        return True

    def NextScpLine(self):
        if self.state == SequentialTableReaderStateType.kHaveRange:
          sefl.range_holder.Clear()
          sefl.state = SequentialTableReaderStateType.kHaveObject
        if self.state != SequentialTableReaderStateType.kHaveScpLine and \
           self.state != SequentialTableReaderStateType.kHaveObject and \
           self.state != SequentialTableReaderStateType.kFileStart:
            LogError('Invalid state \"%s\"' % self.state)
        line = self.script_input.Stream().Readline()
        if line:
            token = line.rstrip().split()
            if len(token) != 2:
                LogError('Invalid line \"%s\"' % line)
            self.key = token[0]
            data_rxfilename = None
            if token[1].endswith(']'):
                LogError('Range specifier support not implemented yet.')
            else:
                data_rxfilename = token[1]
                self.range = None
                filenames_equal = (self.data_rxfilename == data_rxfilename)
                if not filenames_equal:
                    self.data_rxfilename = data_rxfilename
                if self.state == SequentialTableReaderStateType.kHaveObject:
                    if not filenames_equal:
                        self.holder.Clear()
                        self.state = SequentialTableReaderStateType.kHaveScpLine
                else:
                    self.state = SequentialTableReaderStateType.kHaveScpLine
        else:
            self.state = SequentialTableReaderStateType.kEof
            # There is nothing more in the scp file. Might as well close input
            # streams as we don't need them.
            self.script_input.Close()
            if self.data_input.IsOpen():
                self.data_input.Close()
            self.holder.Clear()  # clear the holder if it was nonempty.
            self.range_holder.Clear()  # clear the range holder if it was nonempty.

    def EnsureObjectLoaded(self):
        """Ensures that we have fully loaded any object associated with the current key.

        Returns:
            A boolean variable indicating if the operation is successful.
        """
        if self.state != SequentialTableReaderStateType.kHaveScpLine and \
           self.state != SequentialTableReaderStateType.kHaveObject and \
           self.state != SequentialTableReaderStateType.kHaveRange:
            LogError('Invalid state \"%s\"' % self.state)
        if self.state == SequentialTableReaderStateType.kHaveScpLine:
            success = self.data_input.Open(self.data_rxfilename)
            if not success:
                LogError('Failed to open file \"%s\"' % self.data_rxfilename)
            if self.holder.Read(self.data_input.Stream(),
                                self.data_input.IsBinary()):
                self.state = SequentialTableReaderStateType.kHaveObject
            else:
                LogError('Failed to load object from \"%s\"'
                         % self.data_rxfilename)
        # At this point the state must be either kHaveObject or kHaveRange.
        if self.range:
            LogError('Range specifier support not implemented yet.')
        return True


class SequentialTableReader(object):
    def __init__(self, rspecifier = None, holder_type = HolderType.kNoHolder):
        self.impl = None
        if rspecifier and not self.Open(rspecifier, holder_type):
            LogError('Error contructing reader with rspecifier \"%s\"'
                     % rspecifier)

    def Open(self, rspecifier, holder_type):
        if self.IsOpen():
            if not self.Close():
                LogError('Failed to close previously opened object.')
        # now impl will be None.
        (rspecifier_type, rxfilename, opts) = ClassifyRspecifier(rspecifier)
        if rspecifier_type == RspecifierType.kArchiveRspecifier:
            self.impl = SequentialTableReaderArchiveImpl(holder_type)
        elif rspecifier_type == RspecifierType.kScriptRspecifier:
            self.impl = SequentialTableReaderScriptImpl(holder_type)
        else:
            LogError('Invalid rspecifier \"%s\"' % rspecifier)
        if not self.impl.Open(rspecifier):
            del self.impl
            self.impl = None
            return False
        if self.impl.opts.background:
            LogError('SequentialTableReaderBackgroundImpl not implemented '
                     'yet.')
        return True

    def Done(self):
        self.CheckImpl()
        return self.impl.Done()

    def Key(self):
        self.CheckImpl()
        return self.impl.Key()

    def Value(self):
        self.CheckImpl()
        return self.impl.Value()

    def Next(self):
        self.CheckImpl()
        self.impl.Next()

    def IsOpen(self):
        return self.impl != None

    def Close(self):
        self.CheckImpl()
        success = self.impl.Close()
        del self.impl
        self.impl = None
        return success

    def CheckImpl(self):
        if not self.impl:
            LogError('Trying to use empty SequentialTableReader (perhaps you '
                     'passed the empty string as an argument to a program?).')


class RandomAccessTableReaderStateType(Enum):
    """Enumerations for RandomAccessTableReader state types.
    """
    kUninitialized = 0  # Uninitialized or closed.
    kNoObject = 1  # Do not have object in holder_.
    kHaveObject = 2  # Have object in holder_.
    kEof = 3  # End of file.
    kError = 4  # Some kind of error-state in the reading.


class RandomAccessTableReaderArchiveImplBase(object):
    """Base class for derived implementations such as unsorted/sorted/doubly sorted.
    """
    def __init__(self, holder_type):
        self.input = Input()  # Input object for the archive.
        self.cur_key = None  # Current key (if state == kHaveObject).
        self.type = holder_type  # type of the holder
        self.holder = NewHolderByType(self.type)  # Holds the object we just
                                                  # read (if state == kHaveObject).
        self.rspecifier = None
        self.archive_rxfilename = None
        self.opts = None
        self.state = RandomAccessTableReaderStateType.kUninitialized

    def Open(self, rspecifier):
        if self.state != RandomAccessTableReaderStateType.kUninitialized:
            if not self.Close():
                LogError('Failed to close previous input \"%s\".'
                         % self.rspecifier)
        (rspecifier_type, rxfilename, opts) = ClassifyRspecifier(rspecifier)
        if rspecifier_type != RspecifierType.kArchiveRspecifier:
            LogError('Invalid rspecifier type \"%s\"' % rspecifier_type)
        self.rspecifier = rspecifier
        self.archive_rxfilename = rxfilename
        self.opts = opts
        if self.holder.IsReadInBinary():
            success = self.input.Open(self.archive_rxfilename)
        else:
            success = self.input.OpenTextMode(self.archive_rxfilename)
        if not success:
            self.state = RandomAccessTableReaderStateType.kUninitialized
            LogError('Failed to open stream \"%s\"' % self.archive_rxfilename)
        else:
            self.state = RandomAccessTableReaderStateType.kNoObject
        return True

    def ReadNextObject(self):
        if self.state != RandomAccessTableReaderStateType.kNoObject:
            LogError('Called from the wrong state \"%s\"' % self.state)
        if self.input.Stream().Eof():
            self.state = RandomAccessTableReaderStateType.kEof
            return False
        self.cur_key = ReadToken(self.input.Stream(),
                                 self.input.IsBinary(),
                                 False)
        c = self.input.Stream().Peek(1)
        # We expect a space ' ' after the key. We also allow tab, just so we
        # can read archives generated by scripts that may not be fully aware
        # of how this format works.
        if c != ' ' and c != '\t' and c != '\n':
            LogError('Invalid archive file format: expected space after key '
                     '\"%s\", got character \"%s\" when reading archive \"%s\".'
                     % (self.cur_key, c, self.archive_rxfilename))
        if c != '\n':  # Consume the space or tab.
            self.input.Stream().Read(1)
        binary = InitKaldiInputStream(self.input.Stream())
        if not self.holder.Read(self.input.Stream(), binary):
            self.holder.Clear()
            LogError('Failed to read object from archive \"%s\"'
                     % self.archive_rxfilename)
        self.state = RandomAccessTableReaderStateType.kHaveObject
        return True

    def IsOpen(self):
        if self.state == RandomAccessTableReaderStateType.kEof or \
           self.state == RandomAccessTableReaderStateType.kError or \
           self.state == RandomAccessTableReaderStateType.kHaveObject or \
           self.state == RandomAccessTableReaderStateType.kNoObject:
            return True
        elif self.state == RandomAccessTableReaderStateType.kUninitialized:
            return False
        else:
            LogError('Invalid state \"%s\"' % self.state)

    def CloseInternal(self):
        """Called by the child-class virutal Close() functions, does the shared
        parts of the cleanup.
        """
        if not self.IsOpen():
            LogError('Called twice or otherwise wrongly.')
        if self.input.IsOpen():
            self.input.Close()
        if self.state == RandomAccessTableReaderStateType.kHaveObject:
            self.holder.Clear()
        ans = (self.state != RandomAccessTableReaderStateType.kError)
        self.state = RandomAccessTableReaderStateType.kUninitialized
        if not ans and self.opts.permissive:
            LogWarning('Error state detected closing reader. Ignoring it '
                       'because you specified permissive mode.')
        return ans


class RandomAccessTableReaderUnsortedArchiveImpl(
          RandomAccessTableReaderArchiveImplBase):
    """RandomAccessTableReaderUnsortedArchiveImpl is for random-access reading
    of archives when the user does not specify the sorted (s) option (in this
    case the called-sorted, or "cs" option, is ignored). This is the least
    efficient of the random access archive readers, in general, but it can be as
    efficient as the others, in speed, memory and latency, if the "once" option
    is specified and it happens that the keys of the archive are the same as the
    keys the code is called with (to HasKey() and Value()), and in the same
    order. However, if you ask it for a key that's not present it will have to
    read the archive till the end and store it all in memory.
    """
    def __init__(self, holder_type):
        super(RandomAccessTableReaderUnsortedArchiveImpl, self).__init__(holder_type)
        self.map = {}
        self.to_delete_key = None

    def Close(self):
        for key in self.map.keys():
            self.map.pop(key, None)
        self.map = {}
        return self.CloseInternal()

    def HasKey(self, key):
        self.HandlePendingDelete()
        (success, value) = self.FindKeyInternal(key, False)
        return success

    def Value(self, key):
        self.HandlePendingDelete()
        (success, value) = self.FindKeyInternal(key, True)
        if not success:
            LogError('No such key \"%s\" in archive \"%s\"'
                     % (key, self.archive_rxfilename))
        return value

    def HandlePendingDelete(self):
        if self.to_delete_key:
            self.map.pop(self.to_delete_key, None)
            self.to_delete_key = None

    def FindKeyInternal(self, key, need_value = False):
        """FindKeyInternal() tries to find the key in the dict "self.map"
        If it is not already there, it reads ahead either until it finds the
        key, or until end of file.  If called with need_value == False,
        it assumes it's called from HasKey() and just returns True or False
        and doesn't otherwise have side effects.  If called with need_value ==
        True, it assumes it's called from Value().  Thus, it will crash
        if it cannot find the key.  If it can find it it puts the value in
        return, and if opts_once == true it will mark that element of the
        map to be deleted.

        Args:
            key: The key to find.
            need_value: whether to return corresponding value or not.

        Returns:
            A tuple containing:
                1. A boolean variable indicating if the operation is successful.
                2. The value corresponding to the key and request, None if did
                   not find it or not requested.
        """
        if key in self.map.keys():  # Found in the map...
            if not need_value:  # Called from HasKey()
                return (True, None)
            else:
                value = self.map[key].Value()
                # value won't be needed again, so mark for deletion.
                if self.opts.once:
                    self.to_delete_key = key
                return (True, value)
        while self.state == RandomAccessTableReaderStateType.kNoObject:
            self.ReadNextObject()
            # Successfully read object.
            if self.state == RandomAccessTableReaderStateType.kHaveObject:
                # We are about to transfer ownership of the object in holder_
                # to self.map. Insert it into self.map.
                self.state = RandomAccessTableReaderStateType.kNoObject
                if self.cur_key in self.map.keys():
                    self.holder.Clear()
                    LogError('Duplicate key \"%s\" in archive \"%s\"'
                             % (self.cur_key, self.archive_rxfilename))
                self.map[self.cur_key] = self.holder
                self.holder = NewHolderByType(self.type)
                if self.cur_key == key:
                    if not need_value:  # Called from HasKey()
                        return (True, None)
                    else:  # Called from Value()
                        value = self.map[key].Value()
                        if self.opts.once:
                            self.to_delete_key = key
                        return (True, value)
        return (False, None)  # We read the entire archive (or got to error
                              # state) and didn't find it.


class RandomAccessTableReaderScriptImpl(object):
    """RandomAccessTableReaderScriptImpl is for random-access reading of
    archives when a script file is specified. For simplicity we just read it in
    all in one go, as it's unlikely someone would generate this from a pipe.
    In principle we could read it on-demand as for the archives, but this would
    probably be overkill.
    """
    def __init__(self, holder_type):
        self.input = Input()
        self.opts = None
        self.rspecifier = None
        self.script = None
        self.keys = None
        self.script_rxfilename = None
        self.key = None
        self.type = holder_type
        self.holder = NewHolderByType(self.type)
        self.data_rxfilename = None
        self.last_found = 0
        self.state = SequentialTableReaderStateType.kUninitialized

    def Open(self, rspecifier):
        """Open a reader for the given rspecifier.

        Args:
            rspecifier: The given rspecifier.

        Returns:
            A boolean variable indicating if the operation is successful.
        """
        # You may call Open from states kUninitialized and kError.
        # It may leave the object in any of the states.
        if self.state == RandomAccessTableReaderStateType.kNoObject or \
           self.state == RandomAccessTableReaderStateType.kHaveObject:
            # call Close() yourself to suppress this exception.
            if not self.Close():
                LogError('Error closing previous input, rspecifier was \"%s\"'
                         % self.rspecifier)
        self.rspecifier = rspecifier
        (rspecifier_type, rxfilename, opts) = ClassifyRspecifier(rspecifier)
        self.script_rxfilename = rxfilename
        self.opts = opts
        if rspecifier_type != RspecifierType.kScriptRspecifier:
            LogError('Invalid rspecifier type \"%s\"' % rspecifier_type)

        script_input = Input()
        if not script_input.Open(self.script_rxfilename):
            LogError('Failed opening script file \"%s\"'
                     % self.script_rxfilename)
        if script_input.IsBinary():
            LogError('script file should not be in binary format.')

        script = list()
        while True:
            line = script_input.Stream().Readline()
            if not line:
                break
            token = line.rstrip().split()
            if len(token) != 2:
                LogError('Invalid line \"%s\"' % line)
            script.append((token[0], token[1]))
        self.script = sorted(script, key=itemgetter(0))
        self.keys = [ key for key, _ in self.script ]

        self.state = RandomAccessTableReaderStateType.kNoObject
        self.key = None

        return True

    def IsOpen(self):
        if self.state == RandomAccessTableReaderStateType.kNoObject or \
           self.state == RandomAccessTableReaderStateType.kHaveObject:
            return True
        else:
            return False

    def Close(self):
        if not self.IsOpen():
            LogError('Called on input that was not open.')
        self.input.Close()
        self.holder.Clear()
        self.last_found = 0
        self.script = None
        self.key = None
        self.data_rxfilename = None
        self.state = SequentialTableReaderStateType.kUninitialized
        return True

    def HasKey(self, key):
        preload = self.opts.permissive
        return self.HasKeyInternal(key, preload)

    def Value(self, key):
        if not self.HasKeyInternal(key, True):
            LogError('Could not get item for key = %s' % key)
        return self.holder.Value()

    def HasKeyInternal(self, key, preload):
        if self.state == SequentialTableReaderStateType.kUninitialized or \
           self.state == SequentialTableReaderStateType.kError:
            LogError('Called on RandomAccessTableReader object that is not open.')
        elif self.state == RandomAccessTableReaderStateType.kHaveObject:
            if key == self.key:
                return True
        else:
            pass

        if not self.LookupKey(key):
            return False
        else:
            if not preload:
                return True
            else:
                data_rxfilename = self.script[self.last_found][1]
                if self.state == RandomAccessTableReaderStateType.kHaveObject and \
                   data_rxfilename != self.data_rxfilename:
                    self.state = RandomAccessTableReaderStateType.kNoObject
                    self.holder.Clear()
                self.key = key
                self.data_rxfilename = data_rxfilename
                if self.state == RandomAccessTableReaderStateType.kNoObject:
                    success = self.input.Open(self.data_rxfilename)
                    if not success:
                        LogError('Failed to open file \"%s\"' % self.data_rxfilename)
                        return False
                    else:
                        if self.holder.Read(self.input.Stream(), self.input.IsBinary()):
                            self.state = RandomAccessTableReaderStateType.kHaveObject
                        else:
                            LogError('Failed to load object from \"%s\"' %
                                     self.data_rxfilename)
                            return False
            return True

    def LookupKey(self, key):
        for i in xrange(2):
            if self.last_found < len(self.script) and \
               self.script[self.last_found][0] == key:
                return True
            self.last_found += 1
        self.last_found -= 1

        idx = bisect.bisect(self.keys, key) - 1
        if self.keys[idx] == key:
            self.last_found = idx
            return True
        else:
            return False


class RandomAccessTableReader(object):
    def __init__(self, rspecifier = None, holder_type = HolderType.kNoHolder):
        self.impl = None
        if rspecifier and not self.Open(rspecifier, holder_type):
            LogError('Error contructing reader with rspecifier \"%s\"'
                     % rspecifier)

    def Open(self, rspecifier, holder_type):
        if self.IsOpen():
            LogError('The reader is already open.')
        (rspecifier_type, rxfilename, opts) = ClassifyRspecifier(rspecifier)
        if rspecifier_type == RspecifierType.kArchiveRspecifier:
            if opts.sorted:
                if opts.called_sorted:  # ark,s,cs case
                    self.impl = \
                        RandomAccessTableReaderDSortedArchiveImpl(holder_type)
                else:  # ark,s case
                    self.impl = \
                        RandomAccessTableReaderSortedArchiveImpl(holder_type)
            else:
                self.impl = \
                    RandomAccessTableReaderUnsortedArchiveImpl(holder_type)
        elif rspecifier_type == RspecifierType.kScriptRspecifier:
            self.impl = RandomAccessTableReaderScriptImpl(holder_type)
        else:
            LogError('Invalid rspecifier \"%s\"' % rspecifier)
        if not self.impl.Open(rspecifier):
            del self.impl
            self.impl = None
            return False
        return True

    def IsOpen(self):
        return self.impl != None

    def Close(self):
        self.CheckImpl()
        success = self.impl.Close()
        del self.impl
        self.impl = None
        return success

    def HasKey(self, key):
        self.CheckImpl()
        if not IsToken(key):
            LogError('Invalid key \"%s\"' % key)
        return self.impl.HasKey(key)

    def Value(self, key):
        self.CheckImpl()
        return self.impl.Value(key)

    def CheckImpl(self):
        if not self.impl:
            LogError('Trying to use empty RandomAccessTableReader (perhaps '
                     'you passed the empty string as an argument to a '
                     'program?).')


class TableWriterStateType(Enum):
    """Enumerations for TableWriter state types.
    """
    kUninitialized = 0  # stream is not open
    kOpen = 1  # stream is open
    kWriteError = 2  # stream is open
    kReadScript = 3
    kNotReadScript = 4  # read of script failed.


class TableWriterArchiveImpl(object):
    """The implementation of TableWriter we use when writing directly to an
    archive with no associated scp.
    """
    def __init__(self, holder_type):
        """Initialize the writer for the given holder type.

        Args:
            holder_type: The given holder type.
        """
        self.wspecifier = None
        self.opts = None
        self.archive_wxfilename = None
        self.output = Output()
        self.type = holder_type
        self.state = TableWriterStateType.kUninitialized

    def Open(self, wspecifier):
        """Open a writer for the given wspecifier.

        Args:
            wspecifier: The given wspecifier.

        Returns:
            A boolean variable indicating if the operation is successful.
        """
        if self.state == TableWriterStateType.kUninitialized:
            pass
        elif self.state == TableWriterStateType.kWriteError:
            LogError('Already open with write error.')
        elif self.state == TableWriterStateType.kOpen:
            if not self.Close():
                LogError('Failed closing previously open stream.')
        else:
            LogError('Invalid state \"%s\"' % self.state)
        self.wspecifier = wspecifier
        (wspecifier_type, archive_filename, script_filename, opts) = \
            ClassifyWspecifier(wspecifier)
        self.archive_wxfilename = archive_filename
        self.opts = opts
        if wspecifier_type != WspecifierType.kArchiveWspecifier:
            LogError('Invalid wspecifier type \"%s\"' % wspecifier_type)
        self.output = Output()
        success = self.output.Open(self.archive_wxfilename,
                                   self.opts.binary,
                                   False)
        if not success:
            self.state = TableWriterStateType.kUninitialized
            LogError('Failed to open stream \"%s\"' % self.archive_wxfilename)
        self.state = TableWriterStateType.kOpen
        return True

    def IsOpen(self):
        if self.state == TableWriterStateType.kUninitialized:
            return False
        elif self.state == TableWriterStateType.kOpen or \
            self.state == TableWriterStateType.kWriteError:
            return True
        else:
            LogError('Invalid state \"%s\"' % self.state)

    def Write(self, key, value):
        if self.state == TableWriterStateType.kOpen:
            pass
        elif self.state == TableWriterStateType.kWriteError:
            LogWarning('Attempting to write to invalid stream.')
        elif self.state == TableWriterStateType.kUninitialized:
            LogError('Invalid state \"%s\"' % self.state)
        else:
            LogError('Invalid state \"%s\"' % self.state)
        # state is now kOpen or kWriteError.
        if not IsToken(key):  # e.g. empty string or has spaces...
            LogError('Using invalid key \"%s\"' % key)
        self.output.Stream().Write('%s ' % key)
        if not WriteHolderValueToStream(self.output.Stream(), self.type,
                                        self.opts.binary, value):
            LogWarning('Write failure to \"%s\"' % self.archive_wxfilename)
            self.state = TableWriterStateType.kWriteError
            return False
        # Even if this Write seems to have succeeded, we fail because a previous
        # Write failed and the archive may be corrupted and unreadable.
        if self.state == TableWriterStateType.kWriteError:
            return False
        if self.opts.flush:
            self.Flush()
        return True

    def Flush(self):
        if self.state == TableWriterStateType.kOpen or \
           self.state == TableWriterStateType.kWriteError:
            self.output.Stream().Flush()
        else:
            LogWarning('Called on not-open writer.')

    def Close(self):
        if not self.IsOpen() or not self.output.IsOpen():
            LogError('Called on a stream that was not open. %s, %s'
                     % (self.IsOpen(), self.output.IsOpen()))
        success = self.output.Close()
        if not success:
            LogWarning('Error closing stream: wspecifier is \"%s\"'
                       % self.wspecifier)
            self.state = TableWriterStateType.kUninitialized
            return False
        if self.state == TableWriterStateType.kWriteError:
            LogWarning('Closing writer in error state: wspecifier is \"%s\"'
                       % self.wspecifier)
            self.state = TableWriterStateType.kUninitialized
            return False
        self.state = TableWriterStateType.kUninitialized
        return True


class TableWriterScriptImpl(object):
    # TODO(cfyeh): implement this.
    pass


class TableWriterBothImpl(object):
    # TODO(cfyeh): implement this.
    pass


class TableWriter(object):
    def __init__(self, wspecifier = None, holder_type = HolderType.kNoHolder):
        self.impl = None
        if wspecifier and not self.Open(wspecifier, holder_type):
            LogError('Error contructing writer with wspecifier \"%s\"'
                     % wspecifier)

    def Open(self, wspecifier, holder_type):
        if self.IsOpen():
            if not self.Close():
                LogError('Failed to close previously open writer.')
        (wspecifier_type, archive_filename, script_filename, opts) = \
            ClassifyWspecifier(wspecifier)
        if wspecifier_type == WspecifierType.kBothWspecifier:
            self.impl = TableWriterBothImpl(holder_type)
        elif wspecifier_type == WspecifierType.kArchiveWspecifier:
            self.impl = TableWriterArchiveImpl(holder_type)
        elif wspecifier_type == WspecifierType.kScriptWspecifier:
            self.impl = TableWriterScriptImpl(holder_type)
        else:
            LogError('Invalid wspecifier \"%s\"' % wspecifier)
        if not self.impl.Open(wspecifier):
            del self.impl
            self.impl = None
            return False
        return True

    def IsOpen(self):
        return self.impl != None

    def Close(self):
        self.CheckImpl()
        success = self.impl.Close()
        del self.impl
        self.impl = None
        return success

    def Write(self, key, value):
        self.CheckImpl()
        if not self.impl.Write(key, value):
            LogError('Faile to write to stream.')

    def Flush(self):
        self.CheckImpl()
        self.impl.Flush()

    def CheckImpl(self):
        if not self.impl:
            LogError('Trying to use empty TableWriter (perhaps you passed '
                     'the empty string as an argument to a program?).')


class SequentialBaseFloatMatrixReader(SequentialTableReader):
    """A wrapper for SequentialTableReader(HolderType.kFloatMatrixHolder).
    To make the I/O code more consistent with Kaldi code.
    """
    def __init__(self, rspecifier):
        super(SequentialBaseFloatMatrixReader,
              self).__init__(rspecifier, HolderType.kFloatMatrixHolder)


class SequentialBaseFloatVectorReader(SequentialTableReader):
    """A wrapper for SequentialTableReader(HolderType.kFloatVectorHolder).
    To make the I/O code more consistent with Kaldi code.
    """
    def __init__(self, rspecifier):
        super(SequentialBaseFloatVectorReader,
              self).__init__(rspecifier, HolderType.kFloatVectorHolder)


class SequentialNnetExampleReader(SequentialTableReader):
    """A wrapper for SequentialTableReader(HolderType.kNnetExampleHolder).
    To make the I/O code more consistent with Kaldi code.
    """
    def __init__(self, rspecifier):
        super(SequentialNnetExampleReader,
              self).__init__(rspecifier, HolderType.kNnetExampleHolder)


class RandomAccessFloatVectorReader(RandomAccessTableReader):
    """A wrapper for RandomAccessTableReader(HolderType.kFloatVectorHolder)
    To make the I/O code more consistent with Kaldi code.
    """
    def __init__(self, rspecifier):
        super(RandomAccessFloatVectorReader,
              self).__init__(rspecifier, HolderType.kFloatVectorHolder)


class RandomAccessPosteriorReader(RandomAccessTableReader):
    """A wrapper for RandomAccessTableReader(HolderType.kFloatPosteriorHolder)
    To make the I/O code more consistent with Kaldi code.
    """
    def __init__(self, rspecifier):
        super(RandomAccessPosteriorReader,
              self).__init__(rspecifier, HolderType.kPosteriorHolder)


class RandomAccessInt32VectorReader(RandomAccessTableReader):
    """A wrapper for RandomAccessTableReader(HolderType.kInt32VectorHolder)
    To make the I/O code more consistent with Kaldi code.
    """
    def __init__(self, rspecifier):
        super(RandomAccessInt32VectorReader,
              self).__init__(rspecifier, HolderType.kInt32VectorHolder)


class BaseFloatMatrixWriter(TableWriter):
    """A wrapper for TableWriter(HolderType.kFloatMatrixHolder).
    To make the I/O code more consistent with Kaldi code.
    """
    def __init__(self, wspecifier):
        super(BaseFloatMatrixWriter,
              self).__init__(wspecifier, HolderType.kFloatMatrixHolder)


class BaseFloatVectorWriter(TableWriter):
    """A wrapper for TableWriter(HolderType.kFloatVectorHolder).
    To make the I/O code more consistent with Kaldi code.
    """
    def __init__(self, wspecifier):
        super(BaseFloatVectorWriter,
              self).__init__(wspecifier, HolderType.kFloatVectorHolder)


class Int32VectorWriter(TableWriter):
    """A wrapper for TableWriter(HolderType.kFloatVectorHolder).
    To make the I/O code more consistent with Kaldi code.
    """
    def __init__(self, wspecifier):
        super(Int32VectorWriter,
              self).__init__(wspecifier, HolderType.kInt32VectorHolder)
