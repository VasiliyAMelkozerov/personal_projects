import pandas as pd
import numpy as np
import os

def get_map_data():
    filename = "match_map_stats.csv"
    return pd.read_csv(filename)


    #to be able to feed the information into a machine learning model
    #I have encoded wins and losses in all categories
    #a model does not inherantly know what win or loss is
    #but here the definition changes based off of what team you are asking from
def team_played_encoded(df):
    df["SFS_played"] = (df.team_one_name == "San Francisco Shock") | (df.team_two_name == "San Francisco Shock")
    df["NYE_played"] = (df.team_one_name == "New York Excelsior") | (df.team_two_name == "New York Excelsior")
    df["PF_played"] = (df.team_one_name == "Philadelphia Fusion") | (df.team_two_name == "Philadelphia Fusion")
    df["LAG_played"] = (df.team_one_name == "Los Angeles Gladiators") | (df.team_two_name == "Los Angeles Gladiators")
    df["ShaD_played"] = (df.team_one_name == "Shanghai Dragons") | (df.team_two_name == "Shanghai Dragons")
    df["SeoD_played"] = (df.team_one_name == "Seoul Dynasty") | (df.team_two_name == "Seoul Dynasty")
    df["LAV_played"] = (df.team_one_name == "Los Angeles Valiant") | (df.team_two_name == "Los Angeles Valiant")
    df["DF_played"] = (df.team_one_name == "Dallas Fuel") | (df.team_two_name == "Dallas Fuel")
    df["LS_played"] = (df.team_one_name == "London Spitfire") | (df.team_two_name == "London Spitfire")
    df["BU_played"] = (df.team_one_name == "Boston Uprising") | (df.team_two_name == "Boston Uprising")
    df["AR_played"] = (df.team_one_name == "Atlanta Reign") | (df.team_two_name == "Atlanta Reign")
    df["HO_played"] = (df.team_one_name == "Houston Outlaws") | (df.team_two_name == "Houston Outlaws")
    df["VT_played"] = (df.team_one_name == "Vancouver Titans") | (df.team_two_name == "Vancouver Titans")
    df["HS_played"] = (df.team_one_name == "Hangzhou Spark") | (df.team_two_name == "Hangzhou Spark")
    df["GC_played"] = (df.team_one_name == "Guangzhou Charge") | (df.team_two_name == "Guangzhou Charge")
    df["PE_played"] = (df.team_one_name == "Paris Eternal") | (df.team_two_name == "Paris Eternal")
    df["CH_played"] = (df.team_one_name == "Chengdu Hunters") | (df.team_two_name == "Chengdu Hunters")
    df["FM_played"] = (df.team_one_name == "Florida Mayhem") | (df.team_two_name == "Florida Mayhem")
    df["TD_played"] = (df.team_one_name == "Toronto Defiant") | (df.team_two_name == "Toronto Defiant")
    df["WJ_played"] = (df.team_one_name == "Washington Justice") | (df.team_two_name == "Washington Justice")
    return df

def team_won_encoded(df):
    df["SFS_match_won"] = (df.match_winner == "San Francisco Shock")
    df["NYE_match_won"] = (df.match_winner == "New York Excelsior")
    df["PF_match_won"] = (df.match_winner == "Philadelphia Fusion")
    df["LAG_match_won"] = (df.match_winner == "Los Angeles Gladiators")
    df["ShaD_match_won"] = (df.match_winner == "Shanghai Dragons")
    df["SeoD_match_won"] = (df.match_winner == "Seoul Dynasty")
    df["LAV_match_won"] = (df.match_winner == "Los Angeles Valiant")
    df["DF_match_won"] = (df.match_winner == "Dallas Fuel")
    df["LS_match_won"] = (df.match_winner == "London Spitfire")
    df["BU_match_won"] = (df.match_winner == "Boston Uprising")
    df["AR_match_won"] = (df.match_winner == "Atlanta Reign")
    df["HO_match_won"] = (df.match_winner == "Houston Outlaws")
    df["VT_match_won"] = (df.match_winner == "Vancouver Titans")
    df["HS_match_won"] = (df.match_winner == "Hangzhou Spark")
    df["GC_match_won"] = (df.match_winner == "Guangzhou Charge")
    df["PE_match_won"] = (df.match_winner == "Paris Eternal")
    df["CH_match_won"] = (df.match_winner == "Chengdu Hunters")
    df["FM_match_won"] = (df.match_winner == "Florida Mayhem")
    df["TD_match_won"] = (df.match_winner == "Toronto Defiant")
    df["WJ_match_won"] = (df.match_winner == "Washington Justice")
    return df 

def team_lost_encoded(df):
    df["SFS_match_lost"] = (df.match_winner != "San Francisco Shock")
    df["NYE_match_lost"] = (df.match_winner != "New York Excelsior")
    df["PF_match_lost"] = (df.match_winner != "Philadelphia Fusion")
    df["LAG_match_lost"] = (df.match_winner != "Los Angeles Gladiators")
    df["ShaD_match_lost"] = (df.match_winner != "Shanghai Dragons")
    df["SeoD_match_lost"] = (df.match_winner != "Seoul Dynasty")
    df["LAV_match_lost"] = (df.match_winner != "Los Angeles Valiant")
    df["DF_match_lost"] = (df.match_winner != "Dallas Fuel")
    df["LS_match_lost"] = (df.match_winner != "London Spitfire")
    df["BU_match_lost"] = (df.match_winner != "Boston Uprising")
    df["AR_match_lost"] = (df.match_winner != "Atlanta Reign")
    df["HO_match_lost"] = (df.match_winner != "Houston Outlaws")
    df["VT_match_lost"] = (df.match_winner != "Vancouver Titans")
    df["HS_match_lost"] = (df.match_winner != "Hangzhou Spark")
    df["GC_match_lost"] = (df.match_winner != "Guangzhou Charge")
    df["PE_match_lost"] = (df.match_winner != "Paris Eternal")
    df["CH_match_lost"] = (df.match_winner != "Chengdu Hunters")
    df["FM_match_lost"] = (df.match_winner != "Florida Mayhem")
    df["TD_match_lost"] = (df.match_winner != "Toronto Defiant")
    df["WJ_match_lost"] = (df.match_winner != "Washington Justice")
    return df

def map_won_encoded(df):
    df["SFS_map_won"] = (df.map_winner == "San Francisco Shock")
    df["NYE_map_won"] = (df.map_winner == "New York Excelsior")
    df["PF_map_won"] = (df.map_winner == "Philadelphia Fusion")
    df["LAG_map_won"] = (df.map_winner == "Los Angeles Gladiators")
    df["ShaD_map_won"] = (df.map_winner == "Shanghai Dragons")
    df["SeoD_map_won"] = (df.map_winner == "Seoul Dynasty")
    df["LAV_map_won"] = (df.map_winner == "Los Angeles Valiant")
    df["DF_map_won"] = (df.map_winner == "Dallas Fuel")
    df["LS_map_won"] = (df.map_winner == "London Spitfire")
    df["BU_map_won"] = (df.map_winner == "Boston Uprising")
    df["AR_map_won"] = (df.map_winner == "Atlanta Reign")
    df["HO_map_won"] = (df.map_winner == "Houston Outlaws")
    df["VT_map_won"] = (df.map_winner == "Vancouver Titans")
    df["HS_map_won"] = (df.map_winner == "Hangzhou Spark")
    df["GC_map_won"] = (df.map_winner == "Guangzhou Charge")
    df["PE_map_won"] = (df.map_winner == "Paris Eternal")
    df["CH_map_won"] = (df.map_winner == "Chengdu Hunters")
    df["FM_map_won"] = (df.map_winner == "Florida Mayhem")
    df["TD_map_won"] = (df.map_winner == "Toronto Defiant")
    df["WJ_map_won"] = (df.map_winner == "Washington Justice")
    return df

def map_loss_encoded(df):
    df["SFS_map_lost"] = (df.map_winner != "San Francisco Shock")
    df["NYE_map_lost"] = (df.map_winner != "New York Excelsior")
    df["PF_map_lost"] = (df.map_winner != "Philadelphia Fusion")
    df["LAG_map_lost"] = (df.map_winner != "Los Angeles Gladiators")
    df["ShaD_map_lost"] = (df.map_winner != "Shanghai Dragons")
    df["SeoD_map_lost"] = (df.map_winner != "Seoul Dynasty")
    df["LAV_map_lost"] = (df.map_winner != "Los Angeles Valiant")
    df["DF_map_lost"] = (df.map_winner != "Dallas Fuel")
    df["LS_map_lost"] = (df.map_winner != "London Spitfire")
    df["BU_map_lost"] = (df.map_winner != "Boston Uprising")
    df["AR_map_lost"] = (df.map_winner != "Atlanta Reign")
    df["HO_map_lost"] = (df.map_winner != "Houston Outlaws")
    df["VT_map_lost"] = (df.map_winner != "Vancouver Titans")
    df["HS_map_lost"] = (df.map_winner != "Hangzhou Spark")
    df["GC_map_lost"] = (df.map_winner != "Guangzhou Charge")
    df["PE_map_lost"] = (df.map_winner != "Paris Eternal")
    df["CH_map_lost"] = (df.map_winner != "Chengdu Hunters")
    df["FM_map_lost"] = (df.map_winner != "Florida Mayhem")
    df["TD_map_lost"] = (df.map_winner != "Toronto Defiant")
    df["WJ_map_lost"] = (df.map_winner != "Washington Justice")
    return df

def map_name_encoded(df):
    df["Volskaya"] = (df.map_name == "Volskaya Industries")
    df["Kings_Row"] = (df.map_name == "King's Row")
    df["Anubis"] = (df.map_name == "Temple of Anubis")
    df["Lijiang"] = (df.map_name == "Lijiang Tower")
    df["Ilios"] = (df.map_name == "Ilios")
    df["Hanamura"] = (df.map_name == "Hanamura")
    df["Oasis"] = (df.map_name == "Oasis")
    df["Nepal"] = (df.map_name == "Nepal")
    df["Busan"] = (df.map_name == "Busan")
    df["Gibraltar"] = (df.map_name == "Watchpoint: Gibraltar")
    df["Numbani"] = (df.map_name == "Numbani")
    df["Blizzard World"] = (df.map_name == "Blizzard World")
    df["Eichenwalde"] = (df.map_name == "Eichenwalde")
    df["Dorado"] = (df.map_name == "Dorado")
    df["Junkertown"] = (df.map_name == "Junkertown")
    df["Route_66"] = (df.map_name == "Route 66")
    df["Hollywood"] = (df.map_name == "Hollywood")
    df["Horizon_Lunar_Colony"] = (df.map_name == "Horizon Lunar Colony")
    df["Rialto"] = (df.map_name == "Rialto")
    df["Havana"] = (df.map_name == "Havana")
    df["Paris"] = (df.map_name == "Paris")
    return df

def attack_team_encoded(df):
    df["SFS_Offense"] = (df.attacker == "San Francisco Shock")
    df["NYE_Offense"] = (df.attacker == "New York Excelsior")
    df["PF_Offense"] = (df.attacker == "Philadelphia Fusion")
    df["LAG_Offense"] = (df.attacker == "Los Angeles Gladiators")
    df["ShaD_Offense"] = (df.attacker == "Shanghai Dragons")
    df["SeoD_Offense"] = (df.attacker == "Seoul Dynasty")
    df["LAV_Offense"] = (df.attacker == "Los Angeles Valiant")
    df["DF_Offense"] = (df.attacker == "Dallas Fuel")
    df["LS_Offense"] = (df.attacker == "London Spitfire")
    df["BU_Offense"] = (df.attacker == "Boston Uprising")
    df["AR_Offense"] = (df.attacker == "Atlanta Reign")
    df["HO_Offense"] = (df.attacker == "Houston Outlaws")
    df["VT_Offense"] = (df.attacker == "Vancouver Titans")
    df["HS_Offense"] = (df.attacker == "Hangzhou Spark")
    df["GC_Offense"] = (df.attacker == "Guangzhou Charge")
    df["PE_Offense"] = (df.attacker == "Paris Eternal")
    df["CH_Offense"] = (df.attacker == "Chengdu Hunters")
    df["FM_Offense"] = (df.attacker == "Florida Mayhem")
    df["TD_Offense"] = (df.attacker == "Toronto Defiant")
    df["WJ_Offense"] = (df.attacker == "Washington Justice")
    return df

def defense_team_encoded(df):
    df["SFS_Defence"] = (df.defender == "San Francisco Shock")
    df["NYE_Defence"] = (df.defender == "New York Excelsior")
    df["PF_Defence"] = (df.defender == "Philadelphia Fusion")
    df["LAG_Defence"] = (df.defender == "Los Angeles Gladiators")
    df["ShaD_Defence"] = (df.defender == "Shanghai Dragons")
    df["SeoD_Defence"] = (df.defender == "Seoul Dynasty")
    df["LAV_Defence"] = (df.defender == "Los Angeles Valiant")
    df["DF_Defence"] = (df.defender == "Dallas Fuel")
    df["LS_Defence"] = (df.defender == "London Spitfire")
    df["BU_Defence"] = (df.defender == "Boston Uprising")
    df["AR_Defence"] = (df.defender == "Atlanta Reign")
    df["HO_Defence"] = (df.defender == "Houston Outlaws")
    df["VT_Defence"] = (df.defender == "Vancouver Titans")
    df["HS_Defence"] = (df.defender == "Hangzhou Spark")
    df["GC_Defence"] = (df.defender == "Guangzhou Charge")
    df["PE_Defence"] = (df.defender == "Paris Eternal")
    df["CH_Defence"] = (df.defender == "Chengdu Hunters")
    df["FM_Defence"] = (df.defender == "Florida Mayhem")
    df["TD_Defence"] = (df.defender == "Toronto Defiant")
    df["WJ_Defence"] = (df.defender == "Washington Justice")
    return df

def add_all_encodes(df):
    df = team_played_encoded(df)
    df = team_won_encoded(df)
    df = team_lost_encoded(df)
    df = map_won_encoded(df)
    df = map_loss_encoded(df)
    df = map_name_encoded(df)
    df = attack_team_encoded(df)
    df = defense_team_encoded(df)
    return df

#missing values needing adressing
def fill_NaNs(df):
#a lot of the values needing to be filled are 
#features that do not exist on all maps
    df.fillna(0)
    #Filling with 0s so that ML model can do its job
    return df

#NEED TO SPLIT DATA
#most likely ending up as a time series analysis
#there are 2 differing ideals of how i can do the the split
#with this MK 1 its gonna be a time series split

#this variable is for when I can build for loops to iterate instead having to do these things by hand
teams = ["San Francisco Shock",
"New York Excelsior",
"Philadelphia Fusion",
"Los Angeles Gladiators",
"Shanghai Dragons",
"Seoul Dynasty",
"Los Angeles Valiant",
"Dallas Fuel",
"London Spitfire",
"Boston Uprising",
"Atlanta Reign",
"Houston Outlaws",
"Vancouver Titans",
"Hangzhou Spark",
"Guangzhou Charge",
"Paris Eternal",
"Chengdu Hunters",
"Florida Mayhem",
"Toronto Defiant",
"Washington Justice"]

#AM CURRENTLY BUILDING OUT README MD