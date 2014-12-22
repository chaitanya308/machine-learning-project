import simplejson
import optparse
import csv
import os
import pickle

user_weights_filename = "output_resources/user_weights"
user_weights_filename_txt = "output_resources/user_weights_txt"
user_rating_prediction_filename = "output_resources/user_rating_prediction"
optimal_ratings_filename = "output_resources/computed_ratings_of_restaurants"
mae_filename = "output_resources/mae_values"

def main():

    parser = optparse.OptionParser()
    parser.add_option("-f", dest="reviews_filename", help="Name of the reviews file containing lda topics probability, grouped by user")
    parser.add_option("-l", dest="dont_load_weights", help="Don't load previously calculated user weights from the file. Calculate them freshly.", action='store_true', default=False)
    parser.add_option("-r", dest="restaurants_filename", help="Name of the file containing the reviews info grouped by restaurant.")

    (options, args) = parser.parse_args()

    if not options.reviews_filename:
        parser.error("Provide the name of the reviews file grouped by user through -f option.")

    if not options.restaurants_filename:
        parser.error("Provide the name of the file containing the reviews info grouped by restaurants through -r option.")

    alpha = (0, 0.5, 0.75, 1)

    # if the average weights of users are already computed by previous run, then load them from the file.
    user_weights_dict = {}
    if not options.dont_load_weights:
        if os.path.isfile(user_weights_filename):
            fp = open(user_weights_filename, 'rb')
            user_weights_dict = pickle.load(fp)
            fp.close()

    #Go through each line having a particular user's info, and compute the average weight.
    if not user_weights_dict:
        with open(options.reviews_filename, 'r') as fp:
            # Each line is in the format user_id|review_id1:business_id1:star1:prob1,review_id2:business_id2:star2:prob2...
            for line in fp:
                (user_id, reviews) = line.replace('\n', '').split('|')
                
                # Add the topic probabilities of all the reviews of the user
                topic_weight = 0
                num_reviews = 0
                for review in reviews.split(','):
                    (review_id, business_id, star, prob) = review.split(':')
                    topic_weight += float(prob)
                    num_reviews += 1
                
                # Take average of the topics weight by dividing with number of reviews of the user.
                topic_weight = topic_weight/num_reviews
                
                # calculate final weight of user for each alpha value
                user_weights = []
                for alpha_factor in alpha:
                    weight = topic_weight + (alpha_factor*(1-topic_weight))
                    user_weights.append(weight)
        
                user_weights_dict[user_id] = user_weights
                
        # Dump the user weights into a file for later use by this script.
        with open(user_weights_filename, 'wb') as weights_file:
            pickle.dump(user_weights_dict, weights_file)
            weights_file.close()

        #write into csv file too for manual cross verification
        #format is user_id,weight_for_alpha_0,weight_for_alpha_0.5,weight_for_alpha_0.75
        with open(user_weights_filename_txt, 'w') as rfp:
            writer = csv.writer(rfp, delimiter=',')
            # convert dictionary to list, to be able to write into csv file
            user_list = []
            for item in user_weights_dict.items():
                user_list.append([item[0], item[1][0], item[1][1], item[1][2]])
            writer.writerows(user_list)

    mae_naive = 0
    mae_0 = 0
    mae_0_5 = 0
    mae_0_75 = 0
    num_predicts = 0
    predicted_rows = []
    computed_ratings = []    
    mae_rows = []

    # store values for sample size 100
    mae_naive_100 = 0
    mae_0_100 = 0
    mae_0_5_100 = 0
    mae_0_75_100 = 0
    
    # store values for sample size 1000
    mae_naive_1000 = 0
    mae_0_1000 = 0
    mae_0_5_1000 = 0
    mae_0_75_1000 = 0
    
    # store values for sample size 5000
    mae_naive_5000 = 0
    mae_0_5000 = 0
    mae_0_5_5000 = 0
    mae_0_75_5000 = 0
    
    # store values for sample size 1000
    mae_naive_10000 = 0
    mae_0_10000 = 0
    mae_0_5_10000 = 0
    mae_0_75_10000 = 0
    
    # calculate the optimal ratings of the restaurants
    sample_size = 0
    with open(options.restaurants_filename, 'r') as fp:
        # Each line is in the format restaurant_id|review_id1:user_id1:star1:prob1,review_id2:user_id2:star2:prob2...
        for line in fp:
            (restaurant_id, reviews) = line.replace('\n', '').split('|')
            
            # compute the naive rating and optimal rating.
            naive_rating = 0
            optimal_rating_0 = 0
            optimal_rating_0_denom = 0
            optimal_rating_0_5 = 0
            optimal_rating_0_5_denom = 0
            optimal_rating_0_75 = 0
            optimal_rating_0_75_denom = 0
            num_reviews = 0
            missing_user_info = []

            i = 0
            for review in reviews.split(','):
                i += 1
                (review_id, user_id, star, prob) = review.split(':')
                # capture the first user as missing user, so that his rating would be predicted after calculating optimal ratings.
                if i == 1:
                    missing_user_info.append(user_id)
                    missing_user_info.append(review_id)
                    missing_user_info.append(star)

                naive_rating += float(star)
                user_weights = user_weights_dict[user_id]
                num_reviews += 1

                # Add up values for optimal rating for different values of alpha
                optimal_rating_0 += float(star)*float(user_weights[0])
                optimal_rating_0_denom += float(user_weights[0])
                optimal_rating_0_5 += float(star)*float(user_weights[1])
                optimal_rating_0_5_denom += float(user_weights[1])
                optimal_rating_0_75 += float(star)*float(user_weights[2])
                optimal_rating_0_75_denom += float(user_weights[2])
                    
             
            # calculate the final ratings for each restaurant
            naive_rating = naive_rating/num_reviews
            optimal_rating_0 = optimal_rating_0/optimal_rating_0_denom 
            optimal_rating_0_5 = optimal_rating_0_5/optimal_rating_0_5_denom 
            optimal_rating_0_75 = optimal_rating_0_75/optimal_rating_0_75_denom 
            computed_ratings.append([restaurant_id, naive_rating, optimal_rating_0, optimal_rating_0_5, optimal_rating_0_75])

            # predict the rating for the missing user
            naive_prediction = naive_rating
            user_weights = user_weights_dict[missing_user_info[0]]
            optimal_0_prediction = 0 if float(user_weights[0]) == 0 else optimal_rating_0/float(user_weights[0]) 
            optimal_0_5_prediction = 0 if float(user_weights[1]) == 0 else optimal_rating_0_5/float(user_weights[1])
            optimal_0_75_prediction = 0 if float(user_weights[2]) == 0 else optimal_rating_0_75/float(user_weights[2])
           
            predicted_rows.append([user_id, restaurant_id, star, naive_prediction, optimal_0_prediction, optimal_0_5_prediction, optimal_0_75_prediction])

            # Accumulate the absolute error values. Calculate abosule differences.            
            num_predicts += 1
            actual_rating = float(missing_user_info[2])        
            mae_naive = abs(actual_rating-naive_prediction)
            mae_0 = abs(actual_rating-optimal_0_prediction)
            mae_0_5 = abs(actual_rating - optimal_0_5_prediction)
            mae_0_75 = abs(actual_rating - optimal_0_75_prediction)
            mae_rows.append([user_id, restaurant_id, mae_naive, mae_0, mae_0_5, mae_0_75])
            sample_size += 1

            if sample_size == 100:
                mae_naive_100 = mae_naive
                mae_0_100 = mae_0
                mae_0_5_100 = mae_0_5
                mae_0_75_100 = mae_0_75                

            if sample_size == 1000:
                mae_naive_1000 = mae_naive
                mae_0_1000 = mae_0
                mae_0_5_1000 = mae_0_5
                mae_0_75_1000 = mae_0_75                
            
            if sample_size == 5000:
                mae_naive_5000 = mae_naive
                mae_0_5000 = mae_0
                mae_0_5_5000 = mae_0_5
                mae_0_75_5000 = mae_0_75                
            
            if sample_size == 10000:
                mae_naive_10000 = mae_naive
                mae_0_10000 = mae_0
                mae_0_5_10000 = mae_0_5
                mae_0_75_10000 = mae_0_75                

        # write the computed ratings of restaurants to a file for manual cross verification
        #format is restaurant_id,naive_rating,rating_for_alpha_0,rating_for_alpha_0.5,rating_for_alpha_0.75
        with open(optimal_ratings_filename, 'w') as fp:
            writer = csv.writer(fp, delimiter=',')
            writer.writerows(computed_ratings)

        # write the predictions to a file for manual cross verification
        # format is user_id,restaurant_id,actual_rating,naive_prediction,prediction_for_alpha_0,prediction_for_alpha_0.5,prediction_for_alpha_0.75
        with open(user_rating_prediction_filename, 'w') as fp:
            writer = csv.writer(fp, delimiter=',')
            writer.writerows(predicted_rows)
        
        # write the MAE values of the predicted ratings to a file for manual cross verification
        # format is user_id,restaurant_id,naive_mae,alpha_0_mae,alpha_0.5_mae,alpha_0.75_mae
        with open(mae_filename, 'w') as fp:
            writer = csv.writer(fp, delimiter=',')
            writer.writerows(mae_rows)
    
        # compute and print the average error for sample size 100
        naive_mae_avg = mae_naive_100/100
        optimal_0_mae_avg = mae_0_100/100
        optimal_0_5_mae_avg = mae_0_5_100/100
        optimal_0_75_mae_avg = mae_0_75_100/100

        print("\nMAE values using 100 samples")
        print("Naive method MAE: {}".format(naive_mae_avg))
        print("Our Optimal method MAE for alpha=0: {}".format(optimal_0_mae_avg))
        print("Our Optimal method MAE for alpha=0.5: {}".format(optimal_0_5_mae_avg))
        print("Our Optimal method MAE for alpha=0.75: {}".format(optimal_0_75_mae_avg))
        
        # compute and print the average error for sample size 1000
        naive_mae_avg = mae_naive_1000/1000
        optimal_0_mae_avg = mae_0_1000/1000
        optimal_0_5_mae_avg = mae_0_5_1000/1000
        optimal_0_75_mae_avg = mae_0_75_1000/1000

        print("\nMAE values using 1000 samples")
        print("Naive method MAE: {}".format(naive_mae_avg))
        print("Our Optimal method MAE for alpha=0: {}".format(optimal_0_mae_avg))
        print("Our Optimal method MAE for alpha=0.5: {}".format(optimal_0_5_mae_avg))
        print("Our Optimal method MAE for alpha=0.75: {}".format(optimal_0_75_mae_avg))
        
        # compute and print the average error for sample size 5000
        naive_mae_avg = mae_naive_5000/5000
        optimal_0_mae_avg = mae_0_5000/5000
        optimal_0_5_mae_avg = mae_0_5_5000/5000
        optimal_0_75_mae_avg = mae_0_75_5000/5000

        print("\nMAE values using 5000 samples")
        print("Naive method MAE: {}".format(naive_mae_avg))
        print("Our Optimal method MAE for alpha=0: {}".format(optimal_0_mae_avg))
        print("Our Optimal method MAE for alpha=0.5: {}".format(optimal_0_5_mae_avg))
        print("Our Optimal method MAE for alpha=0.75: {}".format(optimal_0_75_mae_avg))
        
        # compute and print the average error for sample size 10000
        naive_mae_avg = mae_naive_10000/10000
        optimal_0_mae_avg = mae_0_10000/10000
        optimal_0_5_mae_avg = mae_0_5_10000/10000
        optimal_0_75_mae_avg = mae_0_75_10000/10000

        print("\nMAE values using 10000 samples")
        print("Naive method MAE: {}".format(naive_mae_avg))
        print("Our Optimal method MAE for alpha=0: {}".format(optimal_0_mae_avg))
        print("Our Optimal method MAE for alpha=0.5: {}".format(optimal_0_5_mae_avg))
        print("Our Optimal method MAE for alpha=0.75: {}".format(optimal_0_75_mae_avg))
        
        # compute and print the average error
        naive_mae_avg = mae_naive/num_predicts
        optimal_0_mae_avg = mae_0/num_predicts
        optimal_0_5_mae_avg = mae_0_5/num_predicts
        optimal_0_75_mae_avg = mae_0_75/num_predicts

        print("\nMAE values using {} samples".format(num_predicts))
        print("Naive method MAE: {}".format(naive_mae_avg))
        print("Our Optimal method MAE for alpha=0: {}".format(optimal_0_mae_avg))
        print("Our Optimal method MAE for alpha=0.5: {}".format(optimal_0_5_mae_avg))
        print("Our Optimal method MAE for alpha=0.75: {}".format(optimal_0_75_mae_avg))

if __name__  == '__main__':
    main()
