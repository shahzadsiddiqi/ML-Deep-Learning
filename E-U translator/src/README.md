
English to Urdu Translator
This project implements a Machine Learning model to translate text from English to Urdu. It leverages Hugging Face's transformers library and is trained using a dataset available on Hugging Face datasets. The model uses the T5-small architecture fine-tuned specifically for this task.
Features
 Translates English sentences into Urdu.
 Based on a pre-trained T5-small model.
 Fine-tuned on a custom Urdu-English dataset.
 Includes model training, evaluation, and deployment steps.
Dataset
The dataset used for training and testing the model is hosted on Hugging Face: HaiderSultanArc/MT-Urdu-English
Requirements
Install the required Python packages using:
pip install datasets transformers[sentencepiece] sacrebleu tensorflow
Model Architecture
 Base Model: T5-small
 Tokenization: Utilizes AutoTokenizer from Hugging Face.
 Training Configuration:
o Learning Rate: 2e-5 o Batch Size: 16
o Number of Epochs: 2

Code Overview Preprocessing
1. Tokenization:
Prepares the dataset for training and evaluation by tokenizing both English (source) and Urdu (target) texts.
2. Data Reduction:
Selects a subset of the training and testing datasets for faster experimentation.
Training
The model is trained using TensorFlow and the Adam optimizer with weight decay. The data is prepared using DataCollatorForSeq2Seq.
Saving and Loading the Model
The trained model is saved and can be reloaded for future predictions using:
from transformers import TFAutoModelForSeq2SeqLM
model.save_pretrained("english-to-urdu-model")
loaded_model = TFAutoModelForSeq2SeqLM.from_pretrained("english-to-urdu- model")
Prediction
Example usage for translation:
input_text = "hey! how are you."
tokenized = tokenizer([input_text], return_tensors='np') out = loaded_model.generate(**tokenized, max_length=128) print(tokenizer.decode(out[0], skip_special_tokens=True))
Results
The model successfully translates simple sentences from English to Urdu.

How to Use
1. Clone the repository.
2. Install the required dependencies.
3. Run the training script or use the pre-trained model for predictions.
Acknowledgments
 Hugging Face Transformers
 HaiderSultanArc/MT-Urdu-English Dataset
