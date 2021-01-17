import requests
from bs4 import BeautifulSoup
import string
import pandas as pd 
import string
import re
import progressbar
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier # for k nearest neighbors
from sklearn import tree      # for decision trees
from sklearn import ensemble  # for random forests
from sklearn.neural_network import MLPClassifier # for neural nets

try: # different imports for different versions of scikit-learn
    from sklearn.model_selection import cross_val_score   # simpler cv this week
except ImportError:
    try:
        from sklearn.cross_validation import cross_val_score
    except:
        print("No cross_val_score!")

def player_basic_info():
    """returns a pandas data frame of players' basic info from the website
    basketball-reference. The players all started playing after 1989.
    """
    players = []
    base_url = 'http://www.basketball-reference.com/players/'
    for letter in string.ascii_lowercase:
        page_request = requests.get(base_url + letter)
        soup = BeautifulSoup(page_request.text,"lxml")
        table = soup.find('table')
        if table:
            table_body = table.find('tbody')
            for row in table_body.findAll('tr'):
                player_url = row.find('a')
                player_names = player_url.text
                player_pages = player_url['href']
                cells = row.findAll('td') # all data for all players uniform across database
                # print(cells)
                active_from = int(cells[0].text)
                active_to = int(cells[1].text)
                position = cells[2].text
                height = cells[3].text
                weight = cells[4].text
                birth_date = cells[5].text
                college = cells[6].text    
                player_entry = {'url': player_pages,
                                'name': player_names,
                                'active_from': active_from,
                                'active_to': active_to,
                                'position': position,
                                'college': college,
                                'height': height,
                                'weight': weight,
                                'birth_date': birth_date}
                if int(active_from) > 1989:
                    players.append(player_entry)
    return pd.DataFrame(players)
    
def player_info(url):
    """takes in the player-specific url to access their page on 
    basketball-refernce.com. Parses the source code using beautiful soup.
    Collects the career stats and returns it as a dictionary.
    """
    #define all quantites
    rnd = 0
    pick = 0
    g = None
    gs = None
    mpg = None
    fg = None
    fga = None
    fgpct = None
    _3pfg = None
    _3pfga = None
    _3pfgpct = None
    _2pfg = None
    _2pfga = None
    _2pfgpct = None
    efg = None
    ft = None
    fta = None
    ftpct = None
    orb = None
    drb = None
    trb = None
    ast = None
    stl = None
    blk = None
    tov = None
    pf = None
    pts = None

    

    #print('url = ' + str('http://www.basketball-reference.com' + str(url)))
    page_request = requests.get('http://www.basketball-reference.com' + str(url))
    soup = BeautifulSoup(page_request.text,"lxml")
    # print(soup)
    P = soup.findAll("p")
    for p in P:
        # print(p.text)
        if "Draft:" in p.text:
            # print("found it!")
            s = p.text
            # print(s)
            LH1 = s.find("round")
            s1 = s[LH1-7:LH1]
            # print(s1)
            LH2 = s.find("pick")
            RH2 = s.find(")")
            s2 = s[LH2+4:RH2+1]
            # print(s2)
            Round = re.sub('[^0-9]', '', s1)
            # Round = int(Round)
            num = re.sub('[^0-9]', '', s2)
            # num = int(num)
            # print(f"drafted number {num}")
            # print(f"drafted round {Round}")
            rnd = Round
            pick = num

    table = soup.find('table') #the first table is luckily the per game stats
    # print(table)
    # return
    if table:
        # table_foot = table.find('tfoot')
        # print(table_foot)
        for row in table.findAll('tr'):
            headers  = row.findAll('th')
            for h in headers:
                # print(h.text)
                if h.text == 'Career':
                    # print("found it!")
                    cells  = row.findAll('td')
                    # print(cells)
                    List = []
                    for c in cells:
                        # print(c.text.split())
                        # print(c.text)
                        if c.text.split() == []:
                            List += ['']
                        elif c.text.split() == ['NBA']:
                            List += [c.text]
                        elif c.text.split() != []:
                            List += [float(c.text)]
        
    # print(List)
    g = List[4]
    gs = List[5]
    mpg = List[6]
    fg = List[7]
    fga = List[8]
    fgpct = List[9]
    _3pfg = List[10]
    _3pfga = List[11]
    _3pfgpct = List[12]
    _2pfg = List[13]
    _2pfga = List[14]
    _2pfgpct = List[15]
    efg = List[16]
    ft = List[17]
    fta = List[18]
    ftpct = List[19]
    orb = List[20]
    drb = List[21]
    trb = List[22]
    ast = List[23]
    stl = List[24]
    blk = List[25]
    tov = List[26]
    pf = List[27]
    pts = List[28]


    player_entry = {'Draft Round': rnd,'Pick Number': pick,
    'Games Played': g,'Games Started': gs,'Minutes Per Game': mpg,
    'Field Goals Per Game': fg,'Field Goals Attempted Per Game': fga,
    'Field Goal Percentage': fgpct, '3 Point Field Goals Per Game': _3pfg,
    '3 Point Field Goals Attempted Per Game': _3pfga, 
    '3 Point Field Goal Percentage': _3pfgpct,
    '2 Point Field Goals Per Game': _2pfg,
    '2 Point Field Goals Attempted Per Game': _2pfga, 
    '2 Point Field Goal Percentage': _2pfgpct,'Effective Field Goal Percentage': efg,
    'Free Throws Per Game': ft,
    'Free Throws Attempted Per Game': fta, 'Free Throw Percentage': ftpct,
    'Offensive Rebounds Per Game': orb,'Defensive Rebounds Per Game': drb,
    'Total Rebounds Per Game': trb,'Assists Per Game': ast,
    'Steals Per Game': stl,'Blocks Per Game': blk,
    'Turnovers Per Game': tov,'Personal Fouls Per Game': pf,
    'Points Per Game': pts}

    # print('player ' + url + 'complete')
    # print(player_entry)
    return player_entry

def create_csv():
    """Calls player_basic_info(), uses the players' url to call player_info,
    and adds each players career stats to their basic info in the data frame.
    Writes the data frame into a CSV file called players.csv.
    """
    players_general_info = player_basic_info() # call function that scrapes general info
    print('General info/player url loaded...')
    # players_details_info_list = []
    df = pd.DataFrame()	
    bar = progressbar.ProgressBar(maxval=len(players_general_info)).start()
    for i,url in enumerate(players_general_info.url):
        player = player_info(url)
        df = df.append(player, ignore_index = True)
        # print(df)
        bar.update(i)
        # time.sleep(0.1)
    print('Done!') #takes an unholy amount of time
    df = pd.concat([players_general_info, df], axis =1, join="outer")
    df.to_csv('players.csv', encoding='utf-8')


def k_nearest_neighbors_round():
    """Predicts which round a player was drafted in using k nearest 
    neighbors.
    """
    actual_labels = [1,1,1,1,2,2,2,2,0.0,1,1,1,0.0,0.0,1,1,0.0,1,0.0,2]

    df = pd.read_csv('players.csv', header=0)
    
    # df.head()        #first 5 lines
    # df.info()        #column details      

    df = df.drop('index', axis=1)  # axis = 1 indicates we want to drop a column, not a row     
    df = df.drop('url', axis=1)  # axis = 1 indicates we want to drop a column, not a row     
    df = df.drop('active_from', axis=1)  # axis = 1 indicates we want to drop a column, not a row     
    df = df.drop('active_to', axis=1)  # axis = 1 indicates we want to drop a column, not a row     
    df = df.drop('position', axis=1)  # axis = 1 indicates we want to drop a column, not a row     
    df = df.drop('college', axis=1)  # axis = 1 indicates we want to drop a column, not a row     
    df = df.drop('height', axis=1)  # axis = 1 indicates we want to drop a column, not a row     
    df = df.drop('weight', axis=1)  # axis = 1 indicates we want to drop a column, not a row     
    df = df.drop('birth_date', axis=1)  # axis = 1 indicates we want to drop a column, not a row     
    df = df.drop('name', axis=1)  # axis = 1 indicates we want to drop a column, not a row     

    df = df.drop(df.index[170])   #players that were not drafted in round 1, round 2, or undrafted.
    df = df.drop(df.index[603])
    df = df.drop(df.index[739])
    df = df.drop(df.index[768])
    df = df.drop(df.index[774])
    df = df.drop(df.index[1078])
    df = df.drop(df.index[1099])
    df = df.drop(df.index[1306])
    df = df.drop(df.index[1328])
    df = df.drop(df.index[1392])
    df = df.drop(df.index[1430])
    df = df.drop(df.index[1597])
    df = df.drop(df.index[1645])
    df = df.drop(df.index[1793])
    df = df.drop(df.index[1797])
    df = df.drop(df.index[1806])
    df = df.drop(df.index[1861])
    df = df.drop(df.index[2109])
    df = df.drop(df.index[2234])
    



    df = df.dropna()

    # df.head()        #first 5 lines
    # df.info()        #column details  
    
    X1_all_df = df.drop('Draft Round', axis=1)        # everything except the 'Draft Round' column
    X_all_df = X1_all_df.drop('Pick Number', axis=1)        # everything except the 'Pick Number' column
    
    y_all_df = df['Draft Round']                   # the target is Draft Round! 

    feature_names = X_all_df.columns.values.tolist()
    target_names = ["Undrafted", "First Round", "Second Round", "Unknown"]

    X_all = X_all_df.values        
    y_all = y_all_df.values    

    # for i in range(len(y_all)):
    #     if y_all[i] != 0.0 and y_all[i] != 1.0 and y_all[i] != 2.0:
    #         print(i, ',', y_all[i], df.iloc[i, 0])  

    print("+++ start of numpy/scikit-learn +++")

    X_unlabeled = X_all[:20,:]  # unlabeled up to index 20
    y_unlabeled = y_all[:20]    # unlabeled up to index 20

    X_labeled_orig = X_all[20:,:]  # labeled data starts at index 20
    y_labeled_orig = y_all[20:]    # labeled data starts at index 20

    # we scramble the data - but _only_ the labeled data!
    # 
    indices = np.random.permutation(len(y_labeled_orig))  # indices are a permutation

    # we scramble both X and y with the same permutation
    X_labeled = X_labeled_orig[indices]              # we apply the same permutation to each!
    y_labeled = y_labeled_orig[indices]              # again...

    TEST_SIZE = 10
    X_test = X_labeled[:TEST_SIZE]    # first few are for testing
    y_test = y_labeled[:TEST_SIZE]

    X_train = X_labeled[TEST_SIZE:]   # all the rest are for training
    y_train = y_labeled[TEST_SIZE:]

    # averages = {}
    # for k in range(1,51):
    #     knn = KNeighborsClassifier(n_neighbors=k)
    #     cv_scores = cross_val_score( knn, X_train, y_train, cv=5 ) # cv is the number of splits
    #     # print('\nthe cv_scores are')
    #     # for s in cv_scores:
    #         # we format it nicely...
    #         # s_string = "{0:>#7.4f}".format(s) # docs.python.org/3/library/string.html#formatexamples
    #         # print("   ",s_string)
    #     av = cv_scores.mean()
    #     # print('+++ with average: ', av)
    #     print("for k =", k, ", average =", av)
    #     averages[k] = av
    # max_average = max(averages, key=averages.get)
    # print("the k with the best average is", max_average, "and its average is", averages[max_average])

    best_k = 30

    knn_final = KNeighborsClassifier(n_neighbors=best_k)   # now using the best_k
    knn_final.fit(X_labeled, y_labeled)                        # using all of the data
    print("\nCreated and trained a knn classifier with k =", best_k)  #, knn


    print("\n\nFor the input data in X_unlabeled,")
    print("The predicted outputs are:")
    predicted_labels = knn_final.predict(X_unlabeled)
    print(predicted_labels)
    print("\n\n Actual labels are:", actual_labels)

    count = 0
    for i in range(20):
        if predicted_labels[i] == actual_labels[i]:
            count += 1
    
    print("\n\nThe kNN model predicted", count, "out of 20 labels correctly!")

def decision_tree_round():
    """Predicts which round a player was drafted in decision trees.
    """
    actual_labels = [1,1,1,1,2,2,2,2,0.0,1,1,1,0.0,0.0,1,1,0.0,1,0.0,2]

    df = pd.read_csv('players.csv', header=0)
    
    # df.head()        #first 5 lines
    # df.info()        #column details      

    df = df.drop('index', axis=1)  # axis = 1 indicates we want to drop a column, not a row     
    df = df.drop('url', axis=1)  # axis = 1 indicates we want to drop a column, not a row     
    df = df.drop('active_from', axis=1)  # axis = 1 indicates we want to drop a column, not a row     
    df = df.drop('active_to', axis=1)  # axis = 1 indicates we want to drop a column, not a row     
    df = df.drop('position', axis=1)  # axis = 1 indicates we want to drop a column, not a row     
    df = df.drop('college', axis=1)  # axis = 1 indicates we want to drop a column, not a row     
    df = df.drop('height', axis=1)  # axis = 1 indicates we want to drop a column, not a row     
    df = df.drop('weight', axis=1)  # axis = 1 indicates we want to drop a column, not a row     
    df = df.drop('birth_date', axis=1)  # axis = 1 indicates we want to drop a column, not a row     
    df = df.drop('name', axis=1)  # axis = 1 indicates we want to drop a column, not a row     

    df = df.drop(df.index[170])   #players that were not drafted in round 1, round 2, or undrafted.
    df = df.drop(df.index[603])
    df = df.drop(df.index[739])
    df = df.drop(df.index[768])
    df = df.drop(df.index[774])
    df = df.drop(df.index[1078])
    df = df.drop(df.index[1099])
    df = df.drop(df.index[1306])
    df = df.drop(df.index[1328])
    df = df.drop(df.index[1392])
    df = df.drop(df.index[1430])
    df = df.drop(df.index[1597])
    df = df.drop(df.index[1645])
    df = df.drop(df.index[1793])
    df = df.drop(df.index[1797])
    df = df.drop(df.index[1806])
    df = df.drop(df.index[1861])
    df = df.drop(df.index[2109])
    df = df.drop(df.index[2234])
    



    df = df.dropna()

    # df.head()        #first 5 lines
    # df.info()        #column details  
    
    X1_all_df = df.drop('Draft Round', axis=1)        # everything except the 'Draft Round' column
    X_all_df = X1_all_df.drop('Pick Number', axis=1)        # everything except the 'Pick Number' column
    
    y_all_df = df['Draft Round']                   # the target is Draft Round! 

    feature_names = X_all_df.columns.values.tolist()
    target_names = ["Undrafted", "First Round", "Second Round", "Unknown"]

    X_all = X_all_df.values        
    y_all = y_all_df.values    

    # for i in range(len(y_all)):
    #     if y_all[i] != 0.0 and y_all[i] != 1.0 and y_all[i] != 2.0:
    #         print(i, ',', y_all[i], df.iloc[i, 0])  

    print("+++ start of numpy/scikit-learn +++")

    X_unlabeled = X_all[:20,:]  # unlabeled up to index 20
    y_unlabeled = y_all[:20]    # unlabeled up to index 20

    X_labeled_orig = X_all[20:,:]  # labeled data starts at index 20
    y_labeled_orig = y_all[20:]    # labeled data starts at index 20

    # we scramble the data - but _only_ the labeled data!
    # 
    indices = np.random.permutation(len(y_labeled_orig))  # indices are a permutation

    # we scramble both X and y with the same permutation
    X_labeled = X_labeled_orig[indices]              # we apply the same permutation to each!
    y_labeled = y_labeled_orig[indices]              # again...

    TEST_SIZE = 10
    X_test = X_labeled[:TEST_SIZE]    # first few are for testing
    y_test = y_labeled[:TEST_SIZE]

    X_train = X_labeled[TEST_SIZE:]   # all the rest are for training
    y_train = y_labeled[TEST_SIZE:]

    # for max_depth in range(1,6):
    #     # create our classifier
    #     dtree = tree.DecisionTreeClassifier(max_depth=max_depth)
    #     #
    #     # cross-validate to tune our model (this week, all-at-once)
    #     #
    #     scores = cross_val_score(dtree, X_train, y_train, cv=5)
    #     average_cv_score = scores.mean()
    #     # print("      Scores:", scores)
        print("For depth=", max_depth, "average CV score = ", average_cv_score)  
        
    MAX_DEPTH = 3   # choose a MAX_DEPTH based on cross-validation... 
    print("\nChoosing MAX_DEPTH =", MAX_DEPTH, "\n")

    dtree = tree.DecisionTreeClassifier(max_depth=MAX_DEPTH)
    dtree = dtree.fit(X_labeled, y_labeled) 

    #
    # and... Predict the unknown data labels
    #
    print("Decision-tree predictions:\n")
    predicted_labels = dtree.predict(X_unlabeled)

    print("dtree.feature_importances_ are\n      ", dtree.feature_importances_) 
    print("Order:", feature_names[0:4])

    print("\n\n Actual labels are:", actual_labels)
    print("\n\n Predicted labels are:", predicted_labels)

    count = 0
    for i in range(20):
        if predicted_labels[i] == actual_labels[i]:
            count += 1

    print("\n\nThe decision tree model predicted", count, "out of 20 labels correctly!")

def random_forests_round():
    """Predicts which round a player was drafted in using random forests.
    """
    actual_labels = [1,1,1,1,2,2,2,2,0.0,1,1,1,0.0,0.0,1,1,0.0,1,0.0,2]

    df = pd.read_csv('players.csv', header=0)    

    df = df.drop('index', axis=1)  # axis = 1 indicates we want to drop a column, not a row     
    df = df.drop('url', axis=1)  # axis = 1 indicates we want to drop a column, not a row     
    df = df.drop('active_from', axis=1)  # axis = 1 indicates we want to drop a column, not a row     
    df = df.drop('active_to', axis=1)  # axis = 1 indicates we want to drop a column, not a row     
    df = df.drop('position', axis=1)  # axis = 1 indicates we want to drop a column, not a row     
    df = df.drop('college', axis=1)  # axis = 1 indicates we want to drop a column, not a row     
    df = df.drop('height', axis=1)  # axis = 1 indicates we want to drop a column, not a row     
    df = df.drop('weight', axis=1)  # axis = 1 indicates we want to drop a column, not a row     
    df = df.drop('birth_date', axis=1)  # axis = 1 indicates we want to drop a column, not a row     
    df = df.drop('name', axis=1)  # axis = 1 indicates we want to drop a column, not a row     

    df = df.drop(df.index[170])   #players that were not drafted in round 1, round 2, or undrafted.
    df = df.drop(df.index[603])
    df = df.drop(df.index[739])
    df = df.drop(df.index[768])
    df = df.drop(df.index[774])
    df = df.drop(df.index[1078])
    df = df.drop(df.index[1099])
    df = df.drop(df.index[1306])
    df = df.drop(df.index[1328])
    df = df.drop(df.index[1392])
    df = df.drop(df.index[1430])
    df = df.drop(df.index[1597])
    df = df.drop(df.index[1645])
    df = df.drop(df.index[1793])
    df = df.drop(df.index[1797])
    df = df.drop(df.index[1806])
    df = df.drop(df.index[1861])
    df = df.drop(df.index[2109])
    df = df.drop(df.index[2234])
    



    df = df.dropna()

    # df.head()        #first 5 lines
    # df.info()        #column details  
    
    X1_all_df = df.drop('Draft Round', axis=1)        # everything except the 'Draft Round' column
    X_all_df = X1_all_df.drop('Pick Number', axis=1)        # everything except the 'Pick Number' column
    
    y_all_df = df['Draft Round']                   # the target is Draft Round! 

    feature_names = X_all_df.columns.values.tolist()
    target_names = ["Undrafted", "First Round", "Second Round", "Unknown"]

    X_all = X_all_df.values        
    y_all = y_all_df.values    

    # for i in range(len(y_all)):
    #     if y_all[i] != 0.0 and y_all[i] != 1.0 and y_all[i] != 2.0:
    #         print(i, ',', y_all[i], df.iloc[i, 0])  

    print("+++ start of numpy/scikit-learn +++")

    X_unlabeled = X_all[:20,:]  # unlabeled up to index 20
    y_unlabeled = y_all[:20]    # unlabeled up to index 20

    X_labeled_orig = X_all[20:,:]  # labeled data starts at index 20
    y_labeled_orig = y_all[20:]    # labeled data starts at index 20

    # we scramble the data - but _only_ the labeled data!
    # 
    indices = np.random.permutation(len(y_labeled_orig))  # indices are a permutation

    # we scramble both X and y with the same permutation
    X_labeled = X_labeled_orig[indices]              # we apply the same permutation to each!
    y_labeled = y_labeled_orig[indices]              # again...

    TEST_SIZE = 10
    X_test = X_labeled[:TEST_SIZE]    # first few are for testing
    y_test = y_labeled[:TEST_SIZE]

    X_train = X_labeled[TEST_SIZE:]   # all the rest are for training
    y_train = y_labeled[TEST_SIZE:]

    # for m in range(1,7):
    #     for n in range(50, 300, 100):
    #         rforest = ensemble.RandomForestClassifier(max_depth=m, n_estimators=n)

    #         # an example call to run 5x cross-validation on the labeled data
    #         scores = cross_val_score(rforest, X_train, y_train, cv=5)
    #         print(m, n,)
    #         # print("CV scores:", scores)
    #         print("CV scores' average:", scores.mean())

    MAX_DEPTH = 5
    NUM_TREES = 150
    print()
    print("Using MAX_DEPTH=", MAX_DEPTH, "and NUM_TREES=", NUM_TREES)
    rforest = ensemble.RandomForestClassifier(max_depth=MAX_DEPTH, n_estimators=NUM_TREES)
    rforest = rforest.fit(X_labeled, y_labeled) 

    # here are some examples, printed out:
    print("Random-forest predictions:\n")
    predicted_labels = rforest.predict(X_unlabeled)

    print("\n\n Actual labels are:", actual_labels)
    print("\n\n Predicted labels are:", predicted_labels)

    print("\nrforest.feature_importances_ are\n      ", rforest.feature_importances_) 
    print("Order:", feature_names[0:4])

    count = 0
    for i in range(20):
        if predicted_labels[i] == actual_labels[i]:
            count += 1

    print("\n\nThe random forests model predicted", count, "out of 20 labels correctly!")

def neural_nets_rounds():
    """Predicts which round a player was drafted in using neural nets.
    """

    actual_labels = [1,1,1,1,2,2,2,2,0.0,1,1,1,0.0,0.0,1,1,0.0,1,0.0,2]

    df = pd.read_csv('players.csv', header=0)    

    df = df.drop('index', axis=1)  # axis = 1 indicates we want to drop a column, not a row     
    df = df.drop('url', axis=1)  # axis = 1 indicates we want to drop a column, not a row     
    df = df.drop('active_from', axis=1)  # axis = 1 indicates we want to drop a column, not a row     
    df = df.drop('active_to', axis=1)  # axis = 1 indicates we want to drop a column, not a row     
    df = df.drop('position', axis=1)  # axis = 1 indicates we want to drop a column, not a row     
    df = df.drop('college', axis=1)  # axis = 1 indicates we want to drop a column, not a row     
    df = df.drop('height', axis=1)  # axis = 1 indicates we want to drop a column, not a row     
    df = df.drop('weight', axis=1)  # axis = 1 indicates we want to drop a column, not a row     
    df = df.drop('birth_date', axis=1)  # axis = 1 indicates we want to drop a column, not a row     
    df = df.drop('name', axis=1)  # axis = 1 indicates we want to drop a column, not a row     

    df = df.drop(df.index[170])   #players that were not drafted in round 1, round 2, or undrafted.
    df = df.drop(df.index[603])
    df = df.drop(df.index[739])
    df = df.drop(df.index[768])
    df = df.drop(df.index[774])
    df = df.drop(df.index[1078])
    df = df.drop(df.index[1099])
    df = df.drop(df.index[1306])
    df = df.drop(df.index[1328])
    df = df.drop(df.index[1392])
    df = df.drop(df.index[1430])
    df = df.drop(df.index[1597])
    df = df.drop(df.index[1645])
    df = df.drop(df.index[1793])
    df = df.drop(df.index[1797])
    df = df.drop(df.index[1806])
    df = df.drop(df.index[1861])
    df = df.drop(df.index[2109])
    df = df.drop(df.index[2234])
    



    df = df.dropna()

    # df.head()        #first 5 lines
    # df.info()        #column details  
    
    X1_all_df = df.drop('Draft Round', axis=1)        # everything except the 'Draft Round' column
    X_all_df = X1_all_df.drop('Pick Number', axis=1)        # everything except the 'Pick Number' column
    
    y_all_df = df['Draft Round']                   # the target is Draft Round! 

    feature_names = X_all_df.columns.values.tolist()
    target_names = ["Undrafted", "First Round", "Second Round", "Unknown"]

    X_all = X_all_df.values        
    y_all = y_all_df.values    

    print("+++ start of numpy/scikit-learn +++")

    X_unlabeled = X_all[:20,:]  # unlabeled up to index 20
    y_unlabeled = y_all[:20]    # unlabeled up to index 20

    X_labeled_orig = X_all[20:,:]  # labeled data starts at index 20
    y_labeled_orig = y_all[20:]    # labeled data starts at index 20

    # we scramble the data - but _only_ the labeled data!
    # 
    indices = np.random.permutation(len(y_labeled_orig))  # indices are a permutation

    # we scramble both X and y with the same permutation
    X_labeled = X_labeled_orig[indices]              # we apply the same permutation to each!
    y_labeled = y_labeled_orig[indices]              # again...

    TEST_SIZE = 10
    X_test = X_labeled[:TEST_SIZE]    # first few are for testing
    y_test = y_labeled[:TEST_SIZE]

    X_train = X_labeled[TEST_SIZE:]   # all the rest are for training
    y_train = y_labeled[TEST_SIZE:]

    USE_SCALER = True
    if USE_SCALER == True:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(X_train)   # Fit only to the training dataframe
        # now, rescale inputs -- both testing and training
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        X_unlabeled = scaler.transform(X_unlabeled)

    # scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html 
    #

    mlp = MLPClassifier(hidden_layer_sizes=(30,30,30), max_iter=400, alpha=1e-4,
                        solver='sgd', verbose=True, shuffle=True, early_stopping = False, tol=1e-4, 
                        random_state=None, # reproduceability
                        learning_rate_init=.1, learning_rate = 'adaptive')

    print("\n\n++++++++++  TRAINING  +++++++++++++++\n\n")
    mlp.fit(X_train, y_train)


    print("\n\n++++++++++++  TESTING  +++++++++++++\n\n")
    print("Training set score: %f" % mlp.score(X_train, y_train))
    print("Test set score: %f" % mlp.score(X_test, y_test))

    # let's see the coefficients -- the nnet weights!
    # CS = [coef.shape for coef in mlp.coefs_]
    # print(CS)

    # predictions:
    predictions = mlp.predict(X_test)
    from sklearn.metrics import classification_report,confusion_matrix
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test,predictions))

    print("\nClassification report")
    print(classification_report(y_test,predictions))

    predicted_labels = mlp.predict(X_unlabeled)


    print("\n\n Actual labels are:", actual_labels)
    print("\n\n Predicted labels are:", predicted_labels)

    count = 0
    for i in range(20):
        if predicted_labels[i] == actual_labels[i]:
            count += 1

    print("\n\nThe neural network model predicted", count, "out of 20 labels correctly!")
