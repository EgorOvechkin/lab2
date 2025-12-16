# %% [markdown]
# ## Lab 2

# %% [markdown]
# ### Part 3. Poetry generation
# 
# Let's try to generate some poetry using RNNs. 
# 
# You have several choices here: 
# 
# * The Shakespeare sonnets, file `sonnets.txt` available in the notebook directory.
# 
# * Роман в стихах "Евгений Онегин" Александра Сергеевича Пушкина. В предобработанном виде доступен по [ссылке](https://github.com/attatrol/data_sources/blob/master/onegin.txt).
# 
# * Some other text source, if it will be approved by the course staff.
# 
# Text generation can be designed in several steps:
#     
# 1. Data loading.
# 2. Dictionary generation.
# 3. Data preprocessing.
# 4. Model (neural network) training.
# 5. Text generation (model evaluation).
# 

# %%
import string
import os

# %% [markdown]
# ### Data loading: Shakespeare

# %% [markdown]
# Shakespeare sonnets are awailable at this [link](http://www.gutenberg.org/ebooks/1041?msg=welcome_stranger). In addition, they are stored in the same directory as this notebook (`sonnetes.txt`). Simple preprocessing is already done for you in the next cell: all technical info is dropped.

# %%
if not os.path.exists('sonnets.txt'):
    !wget https://raw.githubusercontent.com/girafe-ai/ml-course/22f_basic/homeworks_basic/lab02_deep_learning/sonnets.txt

with open('sonnets.txt', 'r') as iofile:
    text = iofile.readlines()
    
TEXT_START = 45
TEXT_END = -368
text = text[TEXT_START : TEXT_END]
assert len(text) == 2616

# %% [markdown]
# In opposite to the in-class practice, this time we want to predict complex text. Let's reduce the complexity of the task and lowercase all the symbols.
# 
# Now variable `text` is a list of strings. Join all the strings into one and lowercase it.

# %%
# Join all the strings into one and lowercase it
# Put result into variable text.

# Your great code here
text = ''.join(text).lower()

assert len(text) == 100225, 'Are you sure you have concatenated all the strings?'
assert not any([x in set(text) for x in string.ascii_uppercase]), 'Uppercase letters are present'
print('OK!')

# %% [markdown]
# ### Data loading: "Евгений Онегин"
# 

# %%
!curl -L -o onegin.txt https://raw.githubusercontent.com/attatrol/data_sources/master/onegin.txt
    
with open('onegin.txt', 'r', encoding='utf-8') as iofile:
    text = iofile.readlines()
    
text = [x.replace('\t\t', '') for x in text]

# %% [markdown]
# In opposite to the in-class practice, this time we want to predict complex text. Let's reduce the complexity of the task and lowercase all the symbols.
# 
# Now variable `text` is a list of strings. Join all the strings into one and lowercase it.

# %%
# Join all the strings into one and lowercase it
# Put result into variable text.

# Your great code here
text = ''.join(text).lower()

# %% [markdown]
# Put all the characters, that you've seen in the text, into variable `tokens`.

# %%
tokens = sorted(set(out))

# %% [markdown]
# Create dictionary `token_to_idx = {<char>: <index>}` and dictionary `idx_to_token = {<index>: <char>}`

# %%
# dict <index>:<char>
# Your great code here

# dict <char>:<index>
# Your great code here

# %% [markdown]
# *Comment: in this task we have only 38 different tokens, so let's use one-hot encoding.*

# %% [markdown]
# ### Building the model

# %% [markdown]
# Now we want to build and train recurrent neural net which would be able to something similar to Shakespeare's poetry.
# 
# Let's use vanilla RNN, similar to the one created during the lesson.

# %%
# Your code here

# %% [markdown]
# Plot the loss function (axis X: number of epochs, axis Y: loss function).

# %%
# Your plot code here

# %%
def generate_sample(char_rnn, seed_phrase=' Hello', max_length=MAX_LENGTH, temperature=1.0):
    '''
    ### Disclaimer: this is an example function for text generation.
    ### You can either adapt it in your code or create your own function
    
    The function generates text given a phrase of length at least SEQ_LENGTH.
    :param seed_phrase: prefix characters. The RNN is asked to continue the phrase
    :param max_length: maximum output length, including seed_phrase
    :param temperature: coefficient for sampling.  higher temperature produces more chaotic outputs, 
        smaller temperature converges to the single most likely output.
        
    Be careful with the model output. This model waits logits (not probabilities/log-probabilities)
    of the next symbol.
    '''
    
    x_sequence = [token_to_id[token] for token in seed_phrase]
    x_sequence = torch.tensor([[x_sequence]], dtype=torch.int64)
    hid_state = char_rnn.initial_state(batch_size=1)
    
    #feed the seed phrase, if any
    for i in range(len(seed_phrase) - 1):
        print(x_sequence[:, -1].shape, hid_state.shape)
        out, hid_state = char_rnn(x_sequence[:, i], hid_state)
    
    #start generating
    for _ in range(max_length - len(seed_phrase)):
        print(x_sequence.shape, x_sequence, hid_state.shape)
        out, hid_state = char_rnn(x_sequence[:, -1], hid_state)
        # Be really careful here with the model output
        p_next = F.softmax(out / temperature, dim=-1).data.numpy()[0]
        
        # sample next token and push it back into x_sequence
        print(p_next.shape, len(tokens))
        next_ix = np.random.choice(len(tokens), p=p_next)
        next_ix = torch.tensor([[next_ix]], dtype=torch.int64)
        print(x_sequence.shape, next_ix.shape)
        x_sequence = torch.cat([x_sequence, next_ix], dim=1)
        
    return ''.join([tokens[ix] for ix in x_sequence.data.numpy()[0]])

# %%
# An example of generated text.
# print(generate_text(length=500, temperature=0.2))

# %% [markdown]
# ### More poetic model
# 
# Let's use LSTM instead of vanilla RNN and compare the results.

# %% [markdown]
# Plot the loss function of the number of epochs. Does the final loss become better?

# %%
# Your beautiful code here

# %% [markdown]
# Generate text using the trained net with different `temperature` parameter: `[0.1, 0.2, 0.5, 1.0, 2.0]`.
# 
# Evaluate the results visually, try to interpret them.

# %%
# Text generation with different temperature values here

# %% [markdown]
# ### Saving and loading models

# %% [markdown]
# Save the model to the disk, then load it and generate text. Examples are available [here](https://pytorch.org/tutorials/beginner/saving_loading_models.html]).

# %%
# Saving and loading code here

# %% [markdown]
# ### References
# 1. <a href='http://karpathy.github.io/2015/05/21/rnn-effectiveness/'> Andrew Karpathy blog post about RNN. </a> 
# There are several examples of genration: Shakespeare texts, Latex formulas, Linux Sourse Code and children names.
# 2. <a href='https://github.com/karpathy/char-rnn'> Repo with char-rnn code </a>
# 3. Cool repo with PyTorch examples: [link](https://github.com/spro/practical-pytorch`)


