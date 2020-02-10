
import tkinter as tk
import csv
from tqdm import trange

ATTENTION_VISUALIZATION_METHOD = 'custom'  # specify which layer, head, and token yourself
LAYER_ID = 11
HEAD_ID = 11
TOKEN_ID = 0
EXCLUDE_SPECIAL_TOKENS = True  # exclude [CLS] and [SEP] tokens
NUM_EXAMPLES = 100

file_read = open('./token_weights.csv', "rU")
reader = csv.reader(file_read, delimiter=',')



def visualize_attention(window_name, tokens_and_weights):
    """
    Function to visualize the attention mechanism through token highlighting.

    @param (str) window_name: screen-name for the Tkinter window
    @param (list) tokens_and_weights: list containing tuples as elements with tokens and the
           corresponding attention weights, in the form (token, weight); ex. ('awesome', 0.67)
    """
    counter = 0

    root = tk.Tk()
    root.title(window_name)
    text_widget = tk.Text(root)
    text = ''

    # List of indices, where each element will be a tuple in the form: (start_index, end_index)
    low_attention_indices = []
    medium_attention_indices = []
    high_attention_indices = []
    very_high_attention_indices = []

    # Iterate over tokens and weights and assign start and end indices depending on attention weight
    current_index = 0
    for token_and_weight in tokens_and_weights:
        # if counter == 10:
        #     print('hola')
        #     break
        token, weight = token_and_weight[0], token_and_weight[1]
        if token == 'END':
            text += token + ' '
            pass
            counter += 1
            # root = tk.Tk()
            # root.title(weight)
            # text_widget = tk.Text(root)
            #text = weight
        else:
            text += token + ' '
            weight = float(weight)

            if weight >= 0.80:
                very_high_attention_indices.append((current_index, current_index + len(token)))
            elif weight >= 0.60:
                high_attention_indices.append((current_index, current_index + len(token)))
            elif weight >= 0.40:
                medium_attention_indices.append((current_index, current_index + len(token)))
            elif weight >= 0.20:
                low_attention_indices.append((current_index, current_index + len(token)))

        current_index += len(token) + 1

    text_widget.insert(tk.INSERT, text)
    text_widget.pack(expand=1, fill=tk.BOTH)

    # Add Tkinter tags to the specified indices in text widget
    for indices in low_attention_indices:
        text_widget.tag_add('low_attention', '1.' + str(indices[0]), '1.' + str(indices[1]))

    for indices in medium_attention_indices:
        text_widget.tag_add('medium_attention', '1.' + str(indices[0]), '1.' + str(indices[1]))

    for indices in high_attention_indices:
        text_widget.tag_add('high_attention', '1.' + str(indices[0]), '1.' + str(indices[1]))

    for indices in very_high_attention_indices:
        text_widget.tag_add('very_high_attention', '1.' + str(indices[0]), '1.' + str(indices[1]))

    # Highlight attention in text based on defined tags and the corresponding indices
    text_widget.tag_config('low_attention', background='#FDA895')
    text_widget.tag_config('medium_attention', background='#FE7D61')
    text_widget.tag_config('high_attention', background='#FC5430')
    text_widget.tag_config('very_high_attention', background='#FF2D00')

    root.mainloop()

tokens_and_weights = []
for row in reader:
      full_row = [row[0], row[1]]
      tokens_and_weights.append(full_row)
file_read.close()

last_index = 0
for i in trange(NUM_EXAMPLES, desc='Attending to Test Reviews', leave=True):
    new_tokens_and_weights = []
    print('Antes',last_index)
    print(tokens_and_weights[last_index+1])
    clase = tokens_and_weights[last_index][1]
    for j, element in enumerate(tokens_and_weights[last_index+1:]):
        if element[0] == 'END':
            last_index = last_index + (j+1)
            print(last_index)
            break
        else:
            new_tokens_and_weights.append(element)

    visualize_attention(window_name="%s LAYER ID.: %d, HEAD ID.: %d, TOKEN ID.: %d on EXAMPLE ID.: %d" %
                                    (clase,LAYER_ID, HEAD_ID, TOKEN_ID, i),
                        tokens_and_weights=new_tokens_and_weights)
    print('After',last_index)
