machine-learning-project
========================

Project: An optimal rating aggreation mechanism through weighting of aspects extracted from users reviews.

Programming Environment(Linux):
- Download Anaconda 3.4 with Python 3.4.1 from http://continuum.io/downloads
- Setup anaconda environment by following the instructions at http://docs.continuum.io/anaconda/install.html#linux-install
- Activate default environment of Anaconda by running the command 'source $HOME/anaconda3/bin/activate ~/anaconda3'
- Install gensim by following the instructions at http://radimrehurek.com/gensim/install.htmlInstall gensim
    - gensim pre-requisites numpy and scipy also needs to be installed. Use 'pip install pkg_name' to install the dependency packages
- NLTK resources need to be downloaded while running the lda_ml.py program
    - open interactive python window ($ipython)
    - type 'import nltk'
    - type 'nltk.download()'
    - Select the resuorce in the window, and install them

For our project, the environment is setup on CentOs 6.0 and VIM editor is used for source code editing.

Input Dataset:
Yelp challenge dataset available at https://www.yelp.com/dataset_challenge/dataset should be downloaded and unzipped.
The dataset is big, around 1.3GB.

Input data extraction and filtering:
Run the script extract_data to convert the files to CSV and extract relevant data.
$./extract_data

The script has a set of commands to do the following tasks
    - Creates a directory called input_resources
    - Converts yelp_academic_dataset_business.json to input_resources/yelp_academic_dataset_business.csv
    - Converts yelp_academic_dataset_review.json to input_resources/yelp_academic_dataset_review.csv
    - Converts yelp_academic_dataset_user.json to input_resources/yelp_academic_dataset_user.csv
    - Filters out the restaurants from input_resources/yelp_academic_dataset_review.csv into input_resources/restaurant_ids file
    - Filters out users whose review count is greater than 10 into input_resources/users file
    - Filters out the reviews of the restaurants and users whose review count greater than 10 into input_resources/reviews_of_users_with_atleast_11_reviews, and input_resources/reviews_of_users_with_atleast_11_reviews_all_fields

You can directly checkout input_resources directory for the various filtered resource files, instead of running the script.
Please note that all the files passed as inputs to the programs should be present in input_resources directory.

Run the program:
Once all the required input data is present in the input_resources, then proceed to execute the program by running
$./run_rating_aggregation

The script runs lda_ml.py and rating_aggregation.py in sequence to produce final MAE values.

Tasks run by the script are:

- This script would first run the lda_ml.py python program for extraction topics as below. Instead of running the script, lda_ml.py can be invoked directly as below.
$python lda_ml.py -h
Usage: lda_ml.py [options]

Options:
  -h, --help            show this help message and exit
  -f REVIEWS_FILE       reviews filename that contains only text of the
                        review, each in one line.
  -s STOPWORDS_FILE     stopwords filename
  -k NUM_TOPICS         Number of topics
  -d MIN_DF             Minimum document frequency for a token to be included
                        in the dictionary
  -n NUM_TERMS          Number of terms/tokens to be retained in the
                        dictionary
  -w NUM_WORDS          Number of words to be printed for each topic
  -l                    Load previously stored LDA model
  -t TOPIC_INDICES_FILE
                        CSV file containing the numbers of the important
                        topics to be considered
  -c                    If topic probability distribution are available from
                        previous run, then dont capture topic distribution
                        for reviews again
  -r REVIEW_INFO_FILE   CSV file containing the users review information. The
                        order of the reviews in this file and the file
                        specified through -f option should be same.

$python lda_ml.py -f input_resources/reviews_of_users_with_atleast_11_reviews -l -k 15 -w 50 -c -r input_resources/reviews_of_users_with_atleast_11_reviews_all_field    s

This program would output the extracted topics, and the important topic numbers shhould be specified for the program to continue.
Example of input expected is: 1,2,3,4
where the numbers are the topic numbers displayed.

- Once the lda_ml.py program finishes running, it produces the topics probability distribution and saves them in the file output_resources/user_review_topics_total_prob.
- This output file needs to be converted into two different sets of input files for the next program rating_aggregation.py, by running the below two commands
awk -F "," '{gsub("\r","",$5)} {if(a[$2]) a[$2]=a[$2]","$1":"$3":"$4":"$5; else a[$2]=$1":"$3":"$4":"$5;} END {for(i in a) print i,a[i];}' OFS="|" output_resources/    user_review_topics_total_prob > input_resources/reviews_grouped_by_user
awk -F "," '{gsub("\r","",$5)} {if(a[$3]) a[$3]=a[$3]","$1":"$2":"$4":"$5; else a[$3]=$1":"$2":"$4":"$5;} END {for(i in a) print i,a[i];}' OFS="|" output_resources/    user_review_topics_total_prob > input_resources/reviews_grouped_by_restaurant

- Lastly, rating_aggragtion.py is run to calaculate user weights, optimal ratings and MAE values. It is invoked as below
$ python rating_aggregation.py -h
Usage: rating_aggregation.py [options]

Options:
  -h, --help            show this help message and exit
  -f REVIEWS_FILENAME   Name of the reviews file containing lda topics
                        probability, grouped by user
  -l                    Dont load previously calculated user weights from the
                        file. Calculate them freshly.
  -r RESTAURANTS_FILENAME
                        Name of the file containing the reviews info grouped
                        by restaurant.

$python rating_aggregation.py -f input_resources/reviews_grouped_by_user -l -r input_resources/reviews_grouped_by_restaurant

