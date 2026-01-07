################
# DEPENDENCIES #
################

import numpy as np


#############
# CONSTANTS #
#############

COL_PRINT_SPACING = "  "
MAX_N_COLS_PRINT = 10
MAX_N_ROWS_PRINT = 100


##########################
# BASIC HELPER FUNCTIONS #
##########################

def _iter_to_str_arr(lst, pad=" ", min_len=None):
    """Converts a list or 1D numpy array to a 1D numpy array containing only elements of type numpy.str_.
    Each element of the result is a string, optionally padded with 'pad' in which case all elements will have the same length.

    Args:
        lst (list | numpy.ndarray): iterable with elements of types that can be converted meaningfully into strings
        pad (Optional[str | None]): if not None, character to pad elements with untill they reach max('min_len', max(len(elem))), defaults to ' '
        min_len (Optional[None | int]): if not None, minimal length of all elements, ignored if 'pad' is None, defaults to None

    Returns:
        numpy.ndarray: 1D array of numpy.str_ elements
    """
    
    # Checks
    if not(isinstance(lst, list) or isinstance(lst, np.ndarray)):
        raise TypeError(f"'lst' should be of type list or numpy.ndarray, but is of type {type(lst)} instead.")
    if isinstance(lst, np.ndarray):
        if len(lst.shape) != 1:
            raise ValueError(f"'lst' should be 1D, but has shape {lst.shape} instead.")
            
    if not((pad is None) or isinstance(pad, str) or isinstance(pad, np.str_)):
        raise TypeError(f"'pad' should be None or of type str or numpy.str_, but is of type {type(pad)} instead.")
    if (isinstance(pad, str) or isinstance(pad, np.str_)):
        if len(pad) != 1:
            raise ValueError(f"'pad' string should have length 1 but has length {len(pad)} instead.")
            
    if not((min_len is None) or np.issubdtype(type(min_len), np.integer)):
        raise TypeError(f"'min_len' should be None or of type int, but is of type {type(min_len)} instead.")
        
    # All elements to string type
    str_lst = [str(e) for e in lst]

    # Optionally append 'pad' untill all elements are of equal length
    if not(pad is None):
        # Find max_len and pad
        len_lst = [len(e) for e in str_lst]
        max_len = max(len_lst)
        if not(min_len is None):
            max_len = max(max_len, min_len)
        else:
            pass
        fxd_len_str_lst = [e + pad*(max_len-len(e)) for e in str_lst]

        # Conversion to numpy array
        str_arr = np.array(fxd_len_str_lst, dtype=np.str_)
        
    else:
        # Find max_len
        len_lst = [len(e) for e in str_lst]
        max_len = max(len_lst)
        
        # Conversion to numpy array
        str_arr = np.array(str_lst, dtype=np.str_)

    return max_len, str_arr

def _split_idx(idx, n_cols_list, n_arrs, n_cols):
    """Splits index over all columns into an index over the list containing the different arrays and an index within one of those arrays.
    Only works for positive indices, see _split_idx_neg() for negative indices.

    Args:
        idx (int): index to be split
        n_cols_list (list): list of integers containing the number of colums per array
        n_arrs (int): the number of arrays
        n_cols (int): the total number of columns

    Returns:
        int: index over list
        int: index over array
    """
    
    if idx >= n_cols:
        raise IndexError(f"Index {idx} is out of bounds for data with total of {n_cols} columns.")
        
    list_idx = 0
    arr_idx = 0
    cols_passed = 0
    
    while cols_passed < idx:
        if arr_idx == n_cols_list[list_idx]-1:
            list_idx += 1
            arr_idx = 0
        else:
            arr_idx += 1
        cols_passed += 1
        
    return list_idx, arr_idx

def _split_idx_neg(idx, n_cols_list, n_arrs, n_cols):
    """Splits index over all columns into an index over the list containing the different arrays and an index within one of those arrays.
    Only works for negative indices, see _split_idx() for positive indices.

    Args:
        idx (int): index to be split
        n_cols_list (list): list of integers containing the number of colums per array
        n_arrs (int): the number of arrays
        n_cols (int): the total number of columns

    Returns:
        int: index over list
        int: index over array
    """
    
    if idx < -n_cols:
        raise IndexError(f"Index {idx} is out of bound for data with total of {n_cols} columns.")
        
    list_idx = -1
    arr_idx = -1
    cols_passed_neg = -1
    
    while idx < cols_passed_neg:
        if arr_idx == -n_cols_list[list_idx]:
            list_idx -= 1
            arr_idx = -1
        else:
            arr_idx -= 1
        cols_passed_neg -= 1
        
    return list_idx, arr_idx


###########
# CLASSES #
###########

class Table:
    def __init__(self, data, header=None, index=None):
        ### Check input 'data' and initialize related attributes
        # General type check
        if not (isinstance(data, np.ndarray) or isinstance(data, list) or isinstance(data, tuple) or isinstance(data, dict)):
            raise TypeError(f"'data' should be of type numpy.ndarray, list, tuple or dict, but was of type {type(data)} instead.")

        ## Case 1: numpy array
        if isinstance(data, np.ndarray):
            if len(data.shape) != 2:
                raise ValueError(f"'data' should have 2 dimensions, but is of shape {data.shape} instead.")
            if data.size == 0:
                raise ValueError(f"'data' cannot have zero elements.")
                
            self.data = [data]
            self.n_rows, self.n_cols = data.shape
            self.n_cols_list = [data.shape[1]]
            self.n_arrs = 1

        ##  Case 2: list or tuple
        if (isinstance(data, list) or isinstance(data, tuple)):
            # Check if it is not empty
            if len(data) == 0:
                raise ValueError(f"'data' cannot have zero elements.")
            
            # Type and dimensions
            for e in data:
                if not isinstance(e, np.ndarray):
                    raise TypeError(f"If 'data' is a list, its elements should be of type numpy.ndarray, but found element of type {type(e)} instead.")
                if len(e.shape) != 2:
                    raise ValueError(f"If 'data' is a list, its elements should have 2 dimensions, but at least one is of shape {e.shape} instead.")
                if e.size == 0:
                    raise ValueError(f"Arrays in 'data' cannot have zero elements.")

            # Size
            self.n_rows = data[0].shape[0]
            self.n_cols_list = []
            for e in data:
                if e.shape[0] != self.n_rows:
                    raise ValueError(f"All elements of 'data' should have equal length along axis 0, but found lengths {self.n_rows} and {e.shape[0]}.")
                self.n_cols_list.append(e.shape[1])
            self.n_cols = int(np.sum(self.n_cols_list))
            self.n_arrs = len(self.n_cols_list)

            # Assign data
            if isinstance(data, list):
                self.data = data
            if isinstance(data, tuple):
                self.data = [e for e in data]

        ## Case 3: dictionary
        if isinstance(data, dict):
            # Check if it is not empty
            data_keys = data.keys()
            if (data_keys) == 0:
                raise ValueError(f"'data' cannot have zero elements.")
            
            # Type and dimensions
            for key in data_keys:
                if not isinstance(data[key], np.ndarray):
                    raise TypeError(f"If 'data' is a dict, its values should be of type numpy.ndarray, but found value of type {type(data[key])} instead.")
                if len(data[key].shape) != 2:
                    raise ValueError(f"If 'data' is a dict, its values should have 2 dimensions, but at least one is of shape {data[key].shape} instead.")
                if data[key].size == 0:
                    raise ValueError(f"Arrays in 'data' cannot have zero elements.")

            # Size and change to list
            self.n_rows = data[key].shape[0]
            self.n_cols_list = []
            self.data = []
            for key in data_keys:
                if data[key].shape[0] != self.n_rows:
                    raise ValueError(f"All values in 'data' should have equal length along axis 0, but found lengths {self.n_rows} and {data[key].shape[0]}.")
                self.n_cols_list.append(data[key].shape[1])
                self.data.append(data[key])
            self.n_cols = int(np.sum(self.n_cols_list))
            self.n_arrs = len(self.n_cols_list)

            # Header
            header = []
            for i, e in enumerate(data_keys):
                header = header + [e]*self.n_cols_list[i]

        ### Handle header and index
        # Checks on header
        if not (isinstance(header, str) or isinstance(header, np.ndarray) or isinstance(header, list) or isinstance(header, tuple) or (header is None)):
            raise TypeError(f"'header' should be of type str, numpy.ndarray, list, tuple or None, but was of type{type(header)} instead.")
        if isinstance(header, str):
            header = [header]
        if isinstance(header, np.ndarray):
            if len(header.shape) != 1:
                raise ValueError(f"If header is defined as a numpy.ndarray, it should have 1 dimension, but has shape {header.shape} instead.")
        if not (header is None):
            for e in header:
                if not np.isscalar(e):
                    raise TypeError(f"All elements of header must be of a scalar type, found an element of type {type(e)} instead.")

        # Checks on index
        if not (isinstance(index, str) or isinstance(index, np.ndarray) or isinstance(index, list) or isinstance(index, tuple) or (index is None)):
            raise TypeError(f"'index' should be of type str, numpy.ndarray, list, tuple or None, but was of type{type(index)} instead.")
        if isinstance(index, str):
            index = [index]
        if isinstance(index, np.ndarray):
            if len(index.shape) != 1:
                raise ValueError(f"If index is defined as a numpy.ndarray, it should have 1 dimension, but has shape {index.shape} instead.")
        if not (index is None):
            for e in index:
                if not np.isscalar(e):
                    raise TypeError(f"All elements of index must be of a scalar type, found an element of type {type(e)} instead.")

        # Assign header and index stuff
        self.header = header
        self.header_num = np.arange(self.n_cols, dtype=np.int32)

        self.index = index
        self.index_num = np.arange(self.n_rows, dtype=np.int32)

        return

    def __repr__(self):
        ### Sizing up the problem
        # What to print if there are too many columns and/or rows
        filler = "..."
        
        # Logic that is used often -> store in variable
        header_present = not(self.header is None)
        index_present = not(self.index is None)
        too_many_rows = self.n_rows > MAX_N_ROWS_PRINT
        too_many_cols = self.n_cols > MAX_N_COLS_PRINT
        
        # Number of rows and columns to print
        n_rows_print_data = min(MAX_N_ROWS_PRINT, self.n_rows)
        n_rows_print = n_rows_print_data + 1 + header_present + too_many_rows
        n_cols_print_data = min(MAX_N_COLS_PRINT, self.n_cols)
        n_cols_print = n_cols_print_data + 1 + index_present + too_many_cols

        ### Construct columns to print
        ## Index
        if index_present:
            # Find elements to print
            if too_many_rows:
                index_print_1 = [e for e in self.index[:MAX_N_ROWS_PRINT//2]]
                index_print_2 = [e for e in self.index[-MAX_N_ROWS_PRINT//2:]]
                index_print = index_print_1 + [filler] + index_print_2
            else:
                index_print = [e for e in self.index]

            # Add elements to allow header(s)
            index_print.insert(0, " ")
            if header_present:
                index_print.insert(0, " ")
            else:
                pass

            # Conversion
            index_print_width, index_print = _iter_to_str_arr(index_print)
        else:
            pass

        ## Index num
        # Find elements to print
        if too_many_rows:
            index_num_print_1 = [e for e in self.index_num[:MAX_N_ROWS_PRINT//2]]
            index_num_print_2 = [e for e in self.index_num[-MAX_N_ROWS_PRINT//2:]]
            index_num_print = index_num_print_1 + [filler] + index_num_print_2
        else:
            index_num_print = [e for e in self.index_num]

        # Add elements to allow header(s)
        index_num_print.insert(0, " ")
        if header_present:
            index_num_print.insert(0, " ")
        else:
            pass

        # Conversion
        index_num_print_width, index_num_print = _iter_to_str_arr(index_num_print)

        ## Columns with header and data
        col_print_widths = np.zeros(n_cols_print_data+too_many_cols, dtype=np.int32)
        cols_print = []

        if too_many_cols:
            # Add first half of printed columns
            for i in np.arange(MAX_N_COLS_PRINT//2):
                # Split indices
                list_idx, arr_idx = _split_idx(i, self.n_cols_list, self.n_arrs, self.n_cols)
                
                # Find elements to print
                if too_many_rows:
                    col_print_1 = [e for e in self.data[list_idx][:MAX_N_ROWS_PRINT//2, arr_idx]]
                    col_print_2 = [e for e in self.data[list_idx][-MAX_N_ROWS_PRINT//2:, arr_idx]]
                    col_print = col_print_1 + [filler] + col_print_2
                else:
                    col_print = [e for e in self.data[list_idx][:, arr_idx]]
    
                # Add header
                if header_present:
                    col_print.insert(0, self.header[i])
                else:
                    pass
                col_print.insert(0, i)
    
                # Conversion
                col_print_width, col_print = _iter_to_str_arr(col_print)
                
                col_print_widths[i] = col_print_width
                cols_print.append(col_print)

            # Add filler column
            col_print = [filler for i in np.arange(n_rows_print)]
            col_print_width, col_print = _iter_to_str_arr(col_print)
            col_print_widths[MAX_N_COLS_PRINT//2] = col_print_width
            cols_print.append(col_print)

            # Add second half of printed columns
            for i in np.arange(-MAX_N_COLS_PRINT//2, 0):
                # Split indices
                list_idx, arr_idx = _split_idx_neg(i, self.n_cols_list, self.n_arrs, self.n_cols)
                
                # Find elements to print
                if too_many_rows:
                    col_print_1 = [e for e in self.data[list_idx][:MAX_N_ROWS_PRINT//2, arr_idx]]
                    col_print_2 = [e for e in self.data[list_idx][-MAX_N_ROWS_PRINT//2:, arr_idx]]
                    col_print = col_print_1 + [filler] + col_print_2
                else:
                    col_print = [e for e in self.data[list_idx][:, arr_idx]]
    
                # Add header
                if header_present:
                    col_print.insert(0, self.header[i])
                else:
                    pass
                col_print.insert(0, self.header_num[i])
    
                # Conversion
                col_print_width, col_print = _iter_to_str_arr(col_print)
                
                col_print_widths[i] = col_print_width
                cols_print.append(col_print)

        else:
            for i in self.header_num:
                # Split indices
                list_idx, arr_idx = _split_idx(i, self.n_cols_list, self.n_arrs, self.n_cols)
                
                # Find elements to print
                if too_many_rows:
                    col_print_1 = [e for e in self.data[list_idx][:MAX_N_ROWS_PRINT//2, arr_idx]]
                    col_print_2 = [e for e in self.data[list_idx][-MAX_N_ROWS_PRINT//2:, arr_idx]]
                    col_print = col_print_1 + [filler] + col_print_2
                else:
                    col_print = [e for e in self.data[list_idx][:, arr_idx]]
    
                # Add header
                if header_present:
                    col_print.insert(0, self.header[i])
                else:
                    pass
                col_print.insert(0, i)
    
                # Conversion
                col_print_width, col_print = _iter_to_str_arr(col_print)
                
                col_print_widths[i] = col_print_width
                cols_print.append(col_print)

        ## Add index columns to cols_print
        if index_present:
            cols_print.insert(0, index_print)
        else:
            pass
        cols_print.insert(0, index_num_print)

        ### Assemble string
        # Start of print
        res = f"Table with size ({self.n_rows}, {self.n_cols})\n\n"

        # Print per row
        for i in np.arange(n_rows_print):
            row = ""
            for j in np.arange(n_cols_print):
                row = COL_PRINT_SPACING.join((row, cols_print[j][i]))
            row = row[2:] + "\n"
            res = res + row
        
        return res

    def __getitem__(self, idx):
        """This way of indexing only accepts numerical indices. See [...] for indexing with labels.

        Args:
            idx (int | slice | list | numpy.ndarray | tuple): if specified as a tuple, the elements should be of one of the other allowed types.

        Returns
            Table: copy of part of the data, returned as new instance of Table | view of part of the data
        """

        ### Input checks
        # To tuple anyway
        if not isinstance(idx, tuple):
            idx = (idx,)

        # Check dimensions
        n_dim = len(idx)
        if ((n_dim < 1) or (2 < n_dim)):
            raise IndexError(f"Instances of tables.Table have 2 dimensions, but received index of {n_dim} dimensions.")

        # Type checking
        for e in idx:
            if not (isinstance(e, int) or isinstance(e, np.integer) or isinstance(e, slice) or isinstance(e, list) or isinstance(e, np.ndarray)):
                raise IndexError(f"Instances of tables.Table cannot be indexed with objects of type {type(e)}.")

        ## Case 1: only rows are indexed
        if n_dim == 1:
            if isinstance(idx[0], list):
                if len(idx[0]) > self.n_rows:
                    raise IndexError(f"Index (list) in position 0 has {len(idx[0])} elements while the table only has {self.n_rows} rows.")
                for e in idx[0]:
                    if not (isinstance(e, int) or isinstance(e, np.integer) or isinstance(e, bool) or isinstance(e, np.bool)):
                        raise IndexError(f"Index (list) in position 0 has elements of type {type(e)}, index elements must be integers or booleans.")
                    # ...

        # Size checking
        # ...

        ## Case 1: only rows are indexed
        if n_dim == 1:
            print("only rows")

        ## Case 2: rows and columns are indexed
        else:
            print("rows and columns")

        # idx = idx[0]
        # print(idx)
        # print(type(idx))
        # print(isinstance(idx, slice))
        # print("\n")
        # print(idx.start)
        # print(idx.stop)
        # print(idx.step)
        # print("\n")
        # print(arr[idx])

        arr = np.arange(10)
        
        return Table(arr.reshape((2, 5)))