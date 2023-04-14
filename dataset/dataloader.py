import glob2 as glob
import os
import pandas as pd

'''
Listing Fields: [
    id,listing_url,scrape_id,last_scraped,source,name,description,neighborhood_overview,
    picture_url,host_id,host_url,host_name,host_since,host_location,host_about,host_response_time,
    host_response_rate,host_acceptance_rate,host_is_superhost,host_thumbnail_url,host_picture_url,
    host_neighbourhood,host_listings_count,host_total_listings_count,host_verifications,host_has_profile_pic,
    host_identity_verified,neighbourhood,neighbourhood_cleansed,neighbourhood_group_cleansed,latitude,
    longitude,property_type,room_type,accommodates,bathrooms,bathrooms_text,bedrooms,beds,amenities,
    price,minimum_nights,maximum_nights,minimum_minimum_nights,maximum_minimum_nights,minimum_maximum_nights,
    maximum_maximum_nights,minimum_nights_avg_ntm,maximum_nights_avg_ntm,calendar_updated,has_availability,
    availability_30,availability_60,availability_90,availability_365,calendar_last_scraped,number_of_reviews,
    number_of_reviews_ltm,number_of_reviews_l30d,first_review,last_review,review_scores_rating,review_scores_accuracy,
    review_scores_cleanliness,review_scores_checkin,review_scores_communication,review_scores_location,
    review_scores_value,license,instant_bookable,calculated_host_listings_count,calculated_host_listings_count_entire_homes,
    calculated_host_listings_count_private_rooms,calculated_host_listings_count_shared_rooms,reviews_per_month
]
Comments fields : [listing_id,id,date,reviewer_id,reviewer_name,comments]
'''


class Dataloader():
    '''
    The data loader for the AirBnB dataset,
    It expect 2 paths to the (FOLDERS) containing the listing data,
    and the comments datas.
    It loads ALL the csv files in dataframes and provides function to retrive connected data. 
    '''

    def __init__(self, listing_path='./listings', comments_path='./comments'):
        listing_glob_pattner = os.path.join(listing_path, './**/*.csv')
        comments_glob_pattner = os.path.join(comments_path, './**/*.csv')

        listing_files = glob.glob(listing_glob_pattner)
        comments_files = glob.glob(comments_glob_pattner)

        # load listing
        listings = [pd.read_csv(f) for f in listing_files]
        # load listing
        comments = [pd.read_csv(f) for f in comments_files]

        self.listings = pd.concat(listings)
        self.comments = pd.concat(comments)

    def getListings(self):
        return self.listings

    def getComments(self):
        return self.comments

    # Expect dataframe of comments/ single id / array of ids
    def getListingByComments(self, comments):
        if isinstance(comments, pd.DataFrame):
            listings_idx = comments['listing_id']
        elif not isinstance(comments, list):
            listings_idx = [comments]

        return self.listings[self.listings['id'].isin(listings_idx)]

    # Expect dataframe of listings/ single id / array of ids
    def getCommentsFromListing(self, listings):
        if isinstance(listings, pd.DataFrame):
            listings_idx = listings['id']
        elif not isinstance(listings, list):
            listings_idx = [listings]

        return self.comments[self.comments['listing_id'].isin(listings_idx)]
