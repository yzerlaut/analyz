"""
# taken from:
# https://github.com/research-team/Spikes/blob/master/ncs2dat/ncs2dat.py


# Usage: ncs2dat.py input [input...] output
# Converts a list of one-channel Neuralynx NCS files to a single
# multi-channel Neuroscope/KlustaKwik DAT file.
# Channels will be written in the same order as specified
# in the parameters.
"""


import sys

NCS_HEADER_LEN = 16 * 1024
NCS_PAYLOAD_HEADER_LEN = 20
NCS_PAYLOAD_ITEM_LEN = 2
NCS_PAYLOAD_ITEMS = 512
NCS_PAYLOAD_LEN = NCS_PAYLOAD_ITEM_LEN * NCS_PAYLOAD_ITEMS

def main(input_names, output_name,
         max_chunk_number=100000):
    """Execute the script."""
    # Open files and convert, closing them at the end.
    inputs = []
    output = None
    try:
        inputs = [open(input_name, 'rb') for input_name in input_names]
        output = open(output_name, 'wb')
        convert(inputs, output,
                max_chunk_number=max_chunk_number)
    finally:
        for input in inputs:
            input.close()
        if output:
            output.close()


def convert(inputs, output,
            report_after_chunks_count = 1000,
            max_chunk_number=100000):
    """Read data from all inputs and write the data to the output."""
    print('Started')
    count = 0
    # Skip headers.
    for input in inputs:
        input.seek(NCS_HEADER_LEN)
    # Combine chunks from the inputs and write to the output.
    while count<max_chunk_number:
        chunks = [next_chunk(input) for input in inputs]
        if not any(chunks):
            break
        count += 1
        combined = combine(chunks)
        output.write(combined)
        if count % report_after_chunks_count == 0:
            print('Processed {} chunks'.format(count))
    print('Done')


def next_chunk(input):
    """Return the next data chunk from the input, or None at EOF."""
    # Skip record header: assume that the payload size is fixed.
    header = input.read(NCS_PAYLOAD_HEADER_LEN)
    if not header:
        # EOF reached.
        return None
    elif len(header) < NCS_PAYLOAD_HEADER_LEN:
        # Invalid file.
        raise IOError('Invalid record header')
    # The payload is the 512-length array of 16-bit signed integers.
    payload = input.read(NCS_PAYLOAD_LEN)
    return payload


def combine(chunks):
    """Combine a list of chunks for the output."""
    normalize_chunks(chunks)
    combined = bytearray()
    for i in range(0, NCS_PAYLOAD_LEN, NCS_PAYLOAD_ITEM_LEN):
        for chunk in chunks:
            combined.extend(chunk[i : i+NCS_PAYLOAD_ITEM_LEN])
    return combined


def normalize_chunks(chunks):
    """Replace empty chunks with null bytes."""
    for i, chunk in enumerate(chunks):
        if not chunk:
            chunks[i] = bytes(NCS_PAYLOAD_LEN)

            
if __name__ == '__main__':
    
    input_names = ['CSC4.ncs', 'CSC1.ncs', 'CSC8.ncs', 'CSC5.ncs', 'CSC2.ncs', 'CSC3.ncs',
                   'CSC16.ncs', 'CSC13.ncs', 'CSC6.ncs', 'CSC7.ncs', 'CSC12.ncs', 'CSC9.ncs',
                   'CSC10.ncs', 'CSC11.ncs', 'CSC14.ncs', 'CSC15.ncs']
    output_name = 'full.dat'

    main(input_names, output_name)
    
