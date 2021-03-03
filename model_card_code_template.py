import sys


################################
# Complete the functions below #
################################

# Download/create the dataset
def fetch():
  print("fetching dataset!")  # replace this with code to fetch the dataset


# Train your model on the dataset
def train():
  print("training model!")  # replace this with code to train the model


# Compute the evaluation metrics and figures
def evaluate():
  print("evaluating model!")  # replace this with code to evaluate what must be evaluated


# Compile the PDF documents
def build_paper():
  print("building papers!")  # replace this with code to make the papers


###############################
# No need to modify past here #
###############################

supported_functions = {'fetch': fetch,
                       'train': train,
                       'evaluate': evaluate,
                       'build_paper': build_paper}

# If there is no command-line argument, return an error
if len(sys.argv) < 2:
  print("""
    You need to pass in a command-line argument.
    Choose among 'fetch', 'train', 'evaluate' and 'build_paper'.
  """)
  sys.exit(1)

# Extract the first command-line argument, ignoring any others
arg = sys.argv[1]

# Run the corresponding function
if arg in supported_functions:
  supported_functions[arg]()
else:
  raise ValueError("""
    '{}' not among the allowed functions.
    Choose among 'fetch', 'train', 'evaluate' and 'build_paper'.
    """.format(arg))