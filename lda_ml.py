import optparse
import re
import sys
import os

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

import gensim
from gensim import corpora
from gensim import models
from gensim.models import LdaModel

import simplejson
import pickle
import csv

# Various files into which the intermediate data will be stored for use by further pythons scripts & for manual cross verification if reequired
pos_tokens_filename = "output_resources/pos_tokens.txt"
final_tokens_filename = "output_resources/final_tokens.txt"
documents_filename = "output_resources/documents.p"
dictionary_filename = "output_resources/word_dictionary.dict"
dictionary_filename_text = "output_resources/word_dictionary.txt"
corpus_filename = "output_resources/corpus_bow.p"
corpus_filename_text = "output_resources/corpus_bow.txt"
lda_filename = "output_resources/lda_model.lda"
review_topic_prob_filename = "output_resources/user_review_topics_total_prob"
review_all_topic_prob_filename = "output_resources/user_review_individual_topics_prob"

class ReviewLibrary(object):
    def __init__(self, reviews_filename, stopwords_filename):
        self.reviews_filename = reviews_filename
        self.stopwords_filename = stopwords_filename

    def get_stopwords(self):
        f = open(self.stopwords_filename)
        stopword_list = [stopword.strip() for stopword in f]
        f.close()
        return stopword_list

    def get_documents(self):
        # load stopwords, and reviews
        stopword_list = self.get_stopwords()
        reviews_file = open(self.reviews_filename, 'r')
        
        # Dump tokens into file for manual cross-verification
        pos_token_file = open(pos_tokens_filename, 'w')
        final_token_file = open(final_tokens_filename, 'w')

        # process each review in the file to produce words
        documents = []
        i = 0
        for review in reviews_file:
            i += 1
            print("Tokenizing the review {}".format(i))

            # tokenize to get the words and filter out the stop words.
            tokens = [token.lower() for token in nltk.word_tokenize(review) if token.lower() not in stopword_list]

            # Consider only words with text, ignore words with number, special charcter like :) etc         
            text_tokens = [token for token in tokens if re.match('[a-z]+$', token)]
            
            # lemmatize the words to get to the deflated base form, and tag them
            lemmatized_tokens = [WordNetLemmatizer().lemmatize(token) for token in text_tokens]
            tagged_tokens = nltk.pos_tag(lemmatized_tokens)
            simplejson.dump(tagged_tokens, pos_token_file)

            # Consider only those words that represent nouns. Verbs, adjectives & adverbs can also be considered but found that they
            # are yeilding too many unnecessary tokens - (VB*)|(JJ*)|(RB*)
            valid_tokens = [token[0] for token in tagged_tokens if re.match('(NN*)', token[1])]

            # Add the tokens to the documents list
            documents.append(valid_tokens)

        # Dump documents into text file for manual cross-verification in case it is required. 
        simplejson.dump(documents, final_token_file)

        # Serialize the documents into a file, to be used by the script later.
        document_file = open(documents_filename, 'wb')
        pickle.dump(documents, document_file)       

        # Close all the file handles
        final_token_file.close()
        pos_token_file.close()
        reviews_file.close()
        document_file.close()

        self.documents = documents
        return documents

    def get_dictionary(self, documents, min_df, num_terms):
        # Create dictionary from the documents
        dictionary = corpora.Dictionary(documents)
        dictionary.filter_extremes(no_below=min_df, no_above=0.5, keep_n=num_terms)
        dictionary.compactify() 
        
        # Save the dictionary into files for later use
        corpora.Dictionary.save(dictionary, dictionary_filename)
        dictionary.save_as_text(dictionary_filename_text)
        return dictionary
    
def main():
    parser = optparse.OptionParser()
    parser.add_option("-f", dest="reviews_file", help="reviews filename that contains only text of the review, each in one line.")
    parser.add_option("-s", dest="stopwords_file", help="stopwords filename", default="input_resources/stopwords.txt")
    parser.add_option("-k", dest="num_topics", type="int", help="Number of topics", default=20)
    parser.add_option("-d", dest="min_df", type="int", help="Minimum document frequency for a token to be included in the dictionary", default=50)
    parser.add_option("-n", dest="num_terms", type="int", help="Number of terms/tokens to be retained in the dictionary", default=1000000)
    parser.add_option("-w", dest="num_words", type="int", help="Number of words to be printed for each topic", default=20)
    parser.add_option("-l", dest="load_lda", help="Load previously stored LDA model", action='store_true', default=False)
    parser.add_option("-t", dest="topic_indices_file", help="CSV file containing the numbers of the important topics to be considered")
    parser.add_option("-c", dest="skip_capturing_topic_distribution", help="If topic probability distribution are available from previous run, then don't capture topic distribution for reviews again", action='store_true', default=False)
    parser.add_option("-r", dest="review_info_file", help="CSV file containing the user's review information. The order of the reviews in this file and the file specified through -f option should be same.")
    (options, args) = parser.parse_args()

    # Check if the filename of the reviews is provided.
    if not (options.reviews_file):
        parser.error("Please provide name of the reviews file through -f option")
   
    if not (options.review_info_file):
        parser.error("Please provide name of the users' reviews file through -r option")

    print("This program can take several hours to finish based on the size of input file. On yelp dataset, it takes 12 to 15 hours to freshly compute LDA model.")
    print("Please be patient, and allow it to finish.")   
 
    # Extract tokens from reviews, and create a dictionary
    library = ReviewLibrary(options.reviews_file, options.stopwords_file)

    # Try loading the documents from the file first.
    documents = []
    if os.path.isfile(documents_filename):
        document_file = open(documents_filename, 'rb')
        documents = pickle.load(document_file)
        document_file.close()
    # if the contents are not there in the file, then build them.
    if not documents and not options.load_lda and not options.skip_capturing_topic_distribution:
        documents = library.get_documents()        

    # Load the dictionary from the saved file first, if not present, then create the dictionary with the documents i.e. lists of tokens
    dictionary = None
    if os.path.isfile(dictionary_filename):
        dictionary = corpora.Dictionary.load(dictionary_filename)
    if not dictionary and not options.load_lda and not options.skip_capturing_topic_distribution:
        print('Creating dictionary from documents collection...')
        dictionary = library.get_dictionary(documents, options.min_df, options.num_terms)

    # If user asks to use a previously saved LdaModel, then do not run it afresh again.
    lda = None
    if not options.load_lda:
        # Create corpus, try loading it from the file first. If not present in the file, then create the corpus.
        corpus = []
        if os.path.isfile(corpus_filename):
            corpus_file = open(corpus_filename, 'rb')
            corpus = pickle.load(corpus_file)
            corpus_file.close()
        if not corpus:
            print('Creating corpus from documents collection...')
            corpus = [dictionary.doc2bow(document) for document in documents]   

            # Serialize the corpus for later use by the script.
            corpus_file = open(corpus_filename, 'wb')
            pickle.dump(corpus, corpus_file)
            corpus_file.close()

            # Dump into text file for cross verification.
            corpus_file_txt = open(corpus_filename_text, 'w')
            simplejson.dump(corpus, corpus_file_txt) 
            corpus_file_txt.close()

        # Run online LDA
        print('Running online LDA...')
        lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=options.num_topics, update_every=1, chunksize=10000, passes=1)
        lda.save(lda_filename)

    else:
        # load previously saved LdaModel
        lda =  LdaModel.load(lda_filename)
    
    # display the topics
    print("Extracted topics are:\n")
    i = 0
    for topic in lda.show_topics(num_topics=options.num_topics, num_words=options.num_words):
        print("{} : {}".format(i,topic))
        i=i+1;
    
    # get the important topics selected manually
    topic_indices = []
    if options.topic_indices_file:
        topic_indices_file = open(options.topic_indices_file, 'r')
        topic_indices = list(map(int, topic_indices_file.readline().strip().split(',')))
        topic_indices_file.close()

    if not topic_indices:
        # User has not specified imp topics through file, so ask the user to enter the topic numbers
        print("Please enter comma separated numbers of important topics.\n")
        topic_indices = list(map(int, input().split(',')))

    if not options.skip_capturing_topic_distribution:
        review_info_file = open(options.review_info_file, 'r')

        # Iterate over each review document and save the topic probability distributions
        reviews_individual_topic_prob = []
        review_list = []    
        Print('Computing topic probabilities for the review collection...')
        for document,review_line in zip(documents,review_info_file):
            review_info = review_line.strip().split(',')

            # convert the document to bow and pass to lda to get topic distribution
            topics_distr = lda[dictionary.doc2bow(document)]
            reviews_individual_topic_prob.append([review_info[0], topics_distr])                 

            total_prob = 0
            for topic in topics_distr:
                # if the topic is an important one, then consider its prob
                if topic[0] in topic_indices:
                    total_prob += topic[1]
        
            review_info.append(total_prob)
            review_list.append(review_info)

        review_info_file.close()

        # write the total topic probability info of reviews into the file    
        with open(review_topic_prob_filename, 'w') as rfp:
            writer = csv.writer(rfp, delimiter=',')
            writer.writerows(review_list)
        
        # write the individual topic probability info of reviews into the file    
        with open(review_all_topic_prob_filename, 'w') as rfp:
            writer = csv.writer(rfp, delimiter=',')
            writer.writerows(reviews_individual_topic_prob)
       
    
    print("Generated the topic probability distributions for all the reviews")
    print("Saved individual topics probabilities into into {}".format(review_all_topic_prob_filename))
    print("Saved total probability of all important topics for reviews into into {}".format(review_topic_prob_filename))
    print("format of the output {} file is, review_id,user_id,business_id,stars,total_probability_of_all_important_topics. Use this file for calculating user weights".format(review_topic_prob_filename))
 
if __name__ == '__main__':
    main()
