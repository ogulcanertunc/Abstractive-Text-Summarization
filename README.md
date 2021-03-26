<h1 align='center'>ABSTRACTIVE SUMMARIZATION WITH GPT2</h1>

<p align="center">
  <img src="https://github.com/ogulcanertunc/Abstractive-Text-Summarization/blob/main/Images/social-media-1989152_1920.jpg?raw=true" width=600>
</p>

<strong> Here is a demo application of the text summarizer: https://ogisummary.herokuapp.com/ </strong> 
Try it by entering text that you found online or typed yourself; will perceive and summarize the text title and text.

## Business Case

Text Summarization is the task of extracting important information from the original text document. In this process, the extracted information is produced as a report and presented to the user as a short summary. It is very difficult for people to understand and interpret the content of different types of texts, the language structure used and the subject matter are the most important factors in this. In this thesis, our model is set up over a dataset created from news, by addressing abstractive text summarization methods, which are a state-of-the-art. This article collectively summarizes and decipheres the various methodologies, challenges, and problems of abstractive summarization.

The importance of abstractive summary is to save time, as understanding both the abundance of documents and the required and unnecessary documents in many industries is a huge waste of time.



LinkedIn: https://www.linkedin.com/in/ogulcanertunc/ <br>
Medium: https://ertuncogulcan.medium.com/ <br>

## Table of Contents
<details open>
<summary>Show/Hide</summary>
<br>

1. [ File Descriptions ](#File_Description)
2. [ Technologies Used ](#Technologies_Used)    
3. [ Structure ](#Structure)

</details>

## File Descriptions
<details>
<a name="File_Description"></a>
<summary>Show/Hide</summary>
<br>
    
* <strong>[ Data ](https://www.kaggle.com/sunnysai12345/news-summary)</strong>: I found the dataset I used in the project here.
    * <strong>[ Data ](https://github.com/ogulcanertunc/Abstractive-Text-Summarization/tree/main/pt_files)</strong>: You can find the pre-processing datasets in pt format here.
* <strong>(https://github.com/ogulcanertunc/Abstractive-Text-Summarization/tree/main/GPT2_dir)</strong>: training outputs.
* <strong>(https://github.com/ogulcanertunc/Abstractive-Text-Summarization/tree/main/Notebooks)</strong>: notebooks for those who want to work on
* <strong>(./tree/main/Notebooks)</strong>: Project Presentation

</details>

## Tecnologies Used:
<details>
<a name="Technologies_Used"></a>
<summary>Show/Hide</summary>
<br>
    
* <strong>Python</strong>
* <strong>Pandas</strong>
* <strong>Transformers</strong>
* <strong>NLTK</strong>
* <strong>PyTorch</strong>
* <strong>Wrapper</strong>
* <strong>Streamlit</strong>
* <strong>Heroku</strong>
</details>

## Structure of Notebooks:
<details>
<a name="Structure"></a>
<summary>Show/Hide</summary>
<br>
    
1. Pre-Process
   * 1.1 Tokenizing
   * 1.2 Generate Keywords
   * 1.3 Pre-process for GPT Training
   - * 1.3.1 Example after pre-process
   * 1.4 Masking
   - * 1.4.1 Example after masking
   * 1.6 Saving processed Dataset as pt (PyTorch file).

2. Train Model with Masked Data and Summary Creation
   * 2.1 Imports
   * 2.2 Pre-Trained Model Setup (distilgpt2)
   * 2.3 Importing Torch files
   * 2.4 Train a part of train dataset
   - * 2.4.1 Select a row and masked it
   - * 2.4.2 Create a Data Loader
   - * 2.4.3 Learn the shapes of the piece we got from the dataset
   - * 2.4.4 Create Model
   - * 2.4.5 Test Run
   * 2.5 Train
   * 2.6 Saving the Model and Related Files
   * 2.7 Generating Summary
   - * 2.7.1 Determine the Model Directory
   - * 2.7.2 Assign pre-trained models from the Directory
   - * 2.7.3 Activate/Deactivate GPU
   * 2.8 Create a Keywords Maker
   * 2.9 Text/Data Entry
