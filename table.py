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

def iter_to_str_arr(lst, pad=" ", min_len=None):
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


###########
# CLASSES #
###########

class Table:
    def __init__(self, data, header=None, index=None, dtype=None):
        # Checks on data
        if not (isinstance(data, np.ndarray) or isinstance(data, list)):
            raise TypeError(f"'data' should be of type numpy.ndarray, but was of type {type(data)} instead.")

        # Size it up
        self.n_rows, self.n_cols = data.shape

        # Further checks
        # ...

        # Assign header and index stuff
        self.header = header
        self.header_num = np.arange(self.n_cols, dtype=np.int32)

        self.index = index
        self.index_num = np.arange(self.n_rows, dtype=np.int32)

        # Assign data
        self.data = data

        return

    def __repr__(self):
        ### Sizing up the problem
        # What to print is there are too many columns and/or rows
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
            index_print_width, index_print = iter_to_str_arr(index_print)
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
        index_num_print_width, index_num_print = iter_to_str_arr(index_num_print)

        ## Columns with header and data
        col_print_widths = np.zeros(n_cols_print_data+too_many_cols, dtype=np.int32)
        cols_print = []

        if too_many_cols:
            # Add first half of printed columns
            for i in np.arange(MAX_N_COLS_PRINT//2):
                # Find elements to print
                if too_many_rows:
                    col_print_1 = [e for e in self.data[:MAX_N_ROWS_PRINT//2, i]]
                    col_print_2 = [e for e in self.data[-MAX_N_ROWS_PRINT//2:, i]]
                    col_print = col_print_1 + [filler] + col_print_2
                else:
                    col_print = [e for e in self.data[:, i]]
    
                # Add header
                if header_present:
                    col_print.insert(0, self.header[i])
                else:
                    pass
                col_print.insert(0, i)
    
                # Conversion
                col_print_width, col_print = iter_to_str_arr(col_print)
                
                col_print_widths[i] = col_print_width
                cols_print.append(col_print)

            # Add filler column
            col_print = [filler for i in np.arange(n_rows_print)]
            col_print_width, col_print = iter_to_str_arr(col_print)
            col_print_widths[MAX_N_COLS_PRINT//2] = col_print_width
            cols_print.append(col_print)

            # Add second half of printed columns
            for i in np.arange(-MAX_N_COLS_PRINT//2, 0):
                # Find elements to print
                if too_many_rows:
                    col_print_1 = [e for e in self.data[:MAX_N_ROWS_PRINT//2, i]]
                    col_print_2 = [e for e in self.data[-MAX_N_ROWS_PRINT//2:, i]]
                    col_print = col_print_1 + [filler] + col_print_2
                else:
                    col_print = [e for e in self.data[:, i]]
    
                # Add header
                if header_present:
                    col_print.insert(0, self.header[i])
                else:
                    pass
                col_print.insert(0, self.header_num[i])
    
                # Conversion
                col_print_width, col_print = iter_to_str_arr(col_print)
                
                col_print_widths[i] = col_print_width
                cols_print.append(col_print)

        else:
            for i in self.header_num:
                # Find elements to print
                if too_many_rows:
                    col_print_1 = [e for e in self.data[:MAX_N_ROWS_PRINT//2, i]]
                    col_print_2 = [e for e in self.data[-MAX_N_ROWS_PRINT//2:, i]]
                    col_print = col_print_1 + [filler] + col_print_2
                else:
                    col_print = [e for e in self.data[:, i]]
    
                # Add header
                if header_present:
                    col_print.insert(0, self.header[i])
                else:
                    pass
                col_print.insert(0, i)
    
                # Conversion
                col_print_width, col_print = iter_to_str_arr(col_print)
                
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