

def conllu_to_pos(input_file):
    sentences = []
    current_sentence = []
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            line = line.strip()
            if not line or line.startswith('#'):
                continue  # skip empty lines and comments
            columns = line.split('\t')
            if len(columns) > 4:
                word = columns[1]
                pos_tag = columns[4]
                current_sentence.append((word, pos_tag))
            if word == '।':  # End of sentence marker
                sentences.append(current_sentence)
                current_sentence = []

    # Handle the case where the last sentence does not end with '।'
    if current_sentence:
       sentences.append(current_sentence)

    # # Write the result to the output file
    # with open(output_file, 'w', encoding='utf-8') as outfile:
    #     outfile.write(str(sentences))

    return sentences