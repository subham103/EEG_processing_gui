import pandas as pd
import glob
import matplotlib.pyplot as plt
import mne 
import scipy.signal as sps
import scipy.fftpack as spf
import numpy as np
import pywt
import seaborn as sns
import scipy as sp

class0=   'Normal'
Class1 =  'Sick'
###################################################
#file_address = r'D:\EEG\EEG dataset\Epilepic Dataset\E'  # INPUT
###################################################
mne.filter.filter_data

def till_display(file_address):
    
    all_files = glob.glob(file_address + "/*.txt")
    
    li = []
    
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, names=['A'], header=None)
        li.append(df)
    
    file_data = pd.concat(li, axis=0, ignore_index=True)
    
    file_data = file_data.iloc[:,0].values
    
    
    
    ##########################################
    ### Filtering EEG data=> FIR or IIR filters or bandpass filters or None
    ##########################################
    sampling_freq= 173.6  # INPUT
    ##########################################
    apply_filter= 'yes' #['yes','no']  # INPUT
    ############### INPUT
    if(apply_filter=='yes'):
        low_freq= [80, None]
        high_freq= [10, None] # None for low pass filters
        filter_method= ['fir', 'iir']
        window_type=['hamming','hann','blackman']
    
        file_data=mne.filter.filter_data(file_data.astype(float),sfreq=sampling_freq,l_freq=low_freq[0],h_freq=high_freq[1],method=filter_method[0],fir_window=window_type[0])
        
    elif(apply_filter=='no'):
        file_data=file_data
        
    ######################################################################################
    fE,tE,Zxx = sps.stft(x=file_data,fs=sampling_freq)
    
    ZxxE1,ZxxE2,ZxxE3,ZxxE4,ZxxE5,ZxxE6=np.copy(Zxx),np.copy(Zxx),np.copy(Zxx),np.copy(Zxx),np.copy(Zxx),np.copy(Zxx)
    
    ######### EEG Bands ###############################
    
    from numpy import random
    import math
    
    def find_nearest(value):
        b=[]
        for i in range(len(fE)):
           b.append([fE[i],i])
        array=np.array(sorted(b,key=lambda x:x[0]))
        l,r,ans=0,len(array),-1
        while(l<=r):
           mid=l+int((r-l)/2)
           tmp=array[mid][0]
        
           if(tmp<=value):
               ans=mid
               l=mid+1
           else:
               r=mid-1
        if(ans==len(array)-1 or abs(array[ans][0]-value)<abs(array[ans+1][0]-value)):
            return int(array[ans][1])
        else:
            return int(array[ans+1][1])
        
    
    ################################################
    
    ####### INPUT
    delta_freq= [0.5,4]
    theta_freq=  [4,7]
    alpha_freq= [8,12]
    beta_freq=  [12,16]
    gamma_freq= [13,30]
    
    ##### Look into "fE" and get their nearest index number and use the below
    
    delta_in, delta_out= find_nearest(delta_freq[0]),find_nearest(delta_freq[1])
    theta_in, theta_out= find_nearest(theta_freq[0]),find_nearest(theta_freq[1])
    alpha_in, alpha_out= find_nearest(alpha_freq[0]),find_nearest(alpha_freq[1])
    beta_in, beta_out= find_nearest(beta_freq[0]),find_nearest(beta_freq[1])
    gamma_in, gamma_out= find_nearest(gamma_freq[0]),find_nearest(gamma_freq[1])
    
    
    
    
    #Noise
    ZxxE6[:][0:gamma_out+1]= 0
    _,Noise = sps.istft(ZxxE6,fs=173.6)
    #Gamma
    ZxxE1[:][0:gamma_in]= 0
    ZxxE1[:][gamma_out+1:]= 0
    _,Gamma = sps.istft(ZxxE1,fs=173.6)
    #Beta
    ZxxE2[:][0:beta_in]= 0
    ZxxE2[:][beta_out+1:]= 0
    _,Beta = sps.istft(ZxxE2,fs=173.6)
    #Alpha
    ZxxE3[:][0:alpha_in]= 0
    ZxxE3[:][alpha_out+1:]= 0
    _,Alpha = sps.istft(ZxxE3,fs=173.6)
    #Theta
    ZxxE4[:][0:theta_in]= 0
    ZxxE4[:][theta_out+1:]= 0
    _,Theta = sps.istft(ZxxE4,fs=173.6)
    #Delta
    ZxxE5[:][0:delta_in]= 0
    ZxxE5[:][delta_out+1:]= 0
    _,Delta = sps.istft(ZxxE5,fs=173.6)
    
    
    
    
    #############################
    ### Display option for displaying sub-bands  # INPUT
    initial_datapoint= 0    
    final_datapoint= 4097
    
    #########################################
    t= np.arange(0,(23.6/4097)*(final_datapoint-initial_datapoint),1/173.6)
    k=initial_datapoint
    l=final_datapoint
    fig=plt.figure()
    plt.subplots_adjust(bottom=None, top=0.9, hspace=0.8)
    ay = plt.subplot(7,1,1)
    plt.plot(t,file_data[k:l])
    plt.title('Dataset',fontsize=18)
    #plt.axis([None, None, -2000, 2000])
    
    plt.subplot(6,1,2)
    plt.plot(t,Delta[k:l])
    #plt.title('Delta',fontsize=18)
    plt.axis([None, None, -1000, 1000])
    
    
    plt.subplot(6,1,3)
    plt.plot(t,Theta[k:l])
    plt.title('Theta',fontsize=18)
    #plt.axis([None, None, -1000, 1000])
    
    
    plt.subplot(6,1,4)
    plt.plot(t,Alpha[k:l])
    plt.title('Alpha',fontsize=18)
    #plt.axis([None, None, -1000, 1000])
    
    
    plt.subplot(6,1,5)
    plt.plot(t,Beta[k:l])
    plt.title('Beta',fontsize=18)
    #plt.axis([None, None, -1000, 1000])
    
    
    plt.subplot(6,1,6)
    plt.plot(t,Gamma[k:l])
    plt.title('Gamma',fontsize=18)
    #plt.axis([None, None, -200, 200])
    
    
    #plt.subplot(7,1,7)
    #plt.plot(t,Noise[k:l])
    #plt.title('Gamma1')
    #plt.axis([None, None, -200, 200])
    fig.suptitle('Time(sec)',y=0.06,fontsize=16)
    #

#######################################
        
file_address0 = r'D:\EEG\EEG dataset\Epilepic Dataset\A'  # INPUT
file_address1 = r'D:\EEG\EEG dataset\Epilepic Dataset\E'  # INPUT

#### Clasification Task
X_class0= pd.DataFrame(till_display(file_address0))

X_class1= pd.DataFrame(till_display(file_address1))




##########################################################
###########################################################
def till_classification(file_address):
    
    all_files = glob.glob(file_address + "/*.txt")
    
    li = []
    
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, names=['A'], header=None)
        li.append(df)
    
    file_data = pd.concat(li, axis=0, ignore_index=True)
    
    file_data = file_data.iloc[:,0].values
    
    
    
    ##########################################
    ### Filtering EEG data=> FIR or IIR filters or bandpass filters or None
    ##########################################
    sampling_freq= 173.6  # INPUT
    ##########################################
    apply_filter= 'yes' #['yes','no']  # INPUT
    ############### INPUT
    if(apply_filter=='yes'):
        low_freq= [80, None]
        high_freq= [10, None] # None for low pass filters
        filter_method= ['fir', 'iir']
        window_type=['hamming','hann','blackman']
    
        file_data=mne.filter.filter_data(file_data.astype(float),sfreq=sampling_freq,l_freq=low_freq[0],h_freq=high_freq[1],method=filter_method[0],fir_window=window_type[0])
        
    elif(apply_filter=='no'):
        file_data=file_data
        
    ######################################################################################
    
    fE,tE,Zxx = sps.stft(x=file_data,fs=sampling_freq)
    
    ZxxE1,ZxxE2,ZxxE3,ZxxE4,ZxxE5,ZxxE6=np.copy(Zxx),np.copy(Zxx),np.copy(Zxx),np.copy(Zxx),np.copy(Zxx),np.copy(Zxx)
    
    ######### EEG Bands ###############################
    
    from numpy import random
    import math
    
    def find_nearest(value):
        b=[]
        for i in range(len(fE)):
           b.append([fE[i],i])
        array=np.array(sorted(b,key=lambda x:x[0]))
        l,r,ans=0,len(array),-1
        while(l<=r):
           mid=l+int((r-l)/2)
           tmp=array[mid][0]
        
           if(tmp<=value):
               ans=mid
               l=mid+1
           else:
               r=mid-1
        if(ans==len(array)-1 or abs(array[ans][0]-value)<abs(array[ans+1][0]-value)):
            return int(array[ans][1])
        else:
            return int(array[ans+1][1])
        
    
    ################################################
    
    ####### INPUT
    delta_freq= [0.5,4]
    theta_freq=  [4,7]
    alpha_freq= [8,12]
    beta_freq=  [12,16]
    gamma_freq= [13,30]
    
    ##### Look into "fE" and get their nearest index number and use the below
    
    delta_in, delta_out= find_nearest(delta_freq[0]),find_nearest(delta_freq[1])
    theta_in, theta_out= find_nearest(theta_freq[0]),find_nearest(theta_freq[1])
    alpha_in, alpha_out= find_nearest(alpha_freq[0]),find_nearest(alpha_freq[1])
    beta_in, beta_out= find_nearest(beta_freq[0]),find_nearest(beta_freq[1])
    gamma_in, gamma_out= find_nearest(gamma_freq[0]),find_nearest(gamma_freq[1])
    
    
    
    
    #Noise
    ZxxE6[:][0:gamma_out+1]= 0
    _,Noise = sps.istft(ZxxE6,fs=173.6)
    #Gamma
    ZxxE1[:][0:gamma_in]= 0
    ZxxE1[:][gamma_out+1:]= 0
    _,Gamma = sps.istft(ZxxE1,fs=173.6)
    #Beta
    ZxxE2[:][0:beta_in]= 0
    ZxxE2[:][beta_out+1:]= 0
    _,Beta = sps.istft(ZxxE2,fs=173.6)
    #Alpha
    ZxxE3[:][0:alpha_in]= 0
    ZxxE3[:][alpha_out+1:]= 0
    _,Alpha = sps.istft(ZxxE3,fs=173.6)
    #Theta
    ZxxE4[:][0:theta_in]= 0
    ZxxE4[:][theta_out+1:]= 0
    _,Theta = sps.istft(ZxxE4,fs=173.6)
    #Delta
    ZxxE5[:][0:delta_in]= 0
    ZxxE5[:][delta_out+1:]= 0
    _,Delta = sps.istft(ZxxE5,fs=173.6)
    
    
    
    
    #############################
    ### Display option for displaying sub-bands  # INPUT
    initial_datapoint= 0    
    final_datapoint= 4097
    
    #########################################
    t= np.arange(0,(23.6/4097)*(final_datapoint-initial_datapoint),1/173.6)
    k=initial_datapoint
    l=final_datapoint
    fig=plt.figure()
    plt.subplots_adjust(bottom=None, top=0.9, hspace=0.8)
    ay = plt.subplot(7,1,1)
    plt.plot(t,file_data[k:l])
    plt.title('Dataset',fontsize=18)
    #plt.axis([None, None, -2000, 2000])
    
    plt.subplot(6,1,2)
    plt.plot(t,Delta[k:l])
    #plt.title('Delta',fontsize=18)
    plt.axis([None, None, -1000, 1000])
    
    
    plt.subplot(6,1,3)
    plt.plot(t,Theta[k:l])
    plt.title('Theta',fontsize=18)
    #plt.axis([None, None, -1000, 1000])
    
    
    plt.subplot(6,1,4)
    plt.plot(t,Alpha[k:l])
    plt.title('Alpha',fontsize=18)
    #plt.axis([None, None, -1000, 1000])
    
    
    plt.subplot(6,1,5)
    plt.plot(t,Beta[k:l])
    plt.title('Beta',fontsize=18)
    #plt.axis([None, None, -1000, 1000])
    
    
    plt.subplot(6,1,6)
    plt.plot(t,Gamma[k:l])
    plt.title('Gamma',fontsize=18)
    #plt.axis([None, None, -200, 200])
    
    
    #plt.subplot(7,1,7)
    #plt.plot(t,Noise[k:l])
    #plt.title('Gamma1')
    #plt.axis([None, None, -200, 200])
    fig.suptitle('Time(sec)',y=0.06,fontsize=16)
    #
    
    #################################################
    ########## 23.6sec epoch ########################
    
    window_size= 4097 # INPUT
    total_rows= 409700//window_size
    
    ################################################
    
    
    def feature_extraction(method):
        a=[]
        b=[]
        c=[]
        d=[]
        e=[]
        f=[]
        for i in range(total_rows):
            a.append(method(Theta[i*window_size:(window_size*(i+1)-1)]))
        
        for j in range(total_rows):
            b.append(method(Delta[j*window_size:(window_size*(j+1)-1)]))
            
        for k in range(total_rows):
            c.append(method(Alpha[k*window_size:(window_size*(k+1)-1)]))
                
        for l in range(total_rows):
            d.append(method(Beta[l*window_size:(window_size*(l+1)-1)]))
                    
        for m in range(total_rows):
            e.append(method(Gamma[m*window_size:(window_size*(m+1)-1)]))
                      
        for n in range(total_rows):
            f.append(method(Noise[n*window_size:(window_size*(n+1)-1)]))
              
        return a,b,c,d,e,f
    
    
    #################################################
    ######## CORRELATION DIMENSION #######################
    
    def feature_extraction1(method,embedded_dim):
        a=[]
        b=[]
        c=[]
        d=[]
        e=[]
        f=[]
        for i in range(total_rows):
            a.append(method(Theta[i*window_size:(window_size*(i+1)-1)],emb_dim=embedded_dim))
        
        for j in range(total_rows):
            b.append(method(Delta[j*window_size:(window_size*(j+1)-1)],emb_dim=embedded_dim))
            
        for k in range(total_rows):
            c.append(method(Alpha[k*window_size:(window_size*(k+1)-1)],emb_dim=embedded_dim))
                
        for l in range(total_rows):
            d.append(method(Beta[l*window_size:(window_size*(l+1)-1)],emb_dim=embedded_dim))
                    
        for m in range(total_rows):
            e.append(method(Gamma[m*window_size:(window_size*(m+1)-1)],emb_dim=embedded_dim))
                      
        for n in range(total_rows):
            f.append(method(Noise[n*window_size:(window_size*(n+1)-1)],emb_dim=embedded_dim))
              
        return a,b,c,d,e,f
    
     
    ########################################################################
    ################################ Now apply methods   
    # mportant Libraries
    from astropy.stats import median_absolute_deviation
    from numpy import std as standard_dev
    from scipy.integrate import simps as integration
    from scipy.stats import tsem, iqr, kurtosis as trimmed_std, kurtosis
    from numpy import var as variance
    from statistics import mean
    from entropy import app_entropy
    from nolds import hurst_rs  as hurst_expo
    from nolds import dfa as detrended_fluctuation
    
    from nolds import sampen as sample_entropy
    from nolds import corr_dim as correlation_dim
    from nolds import lyap_e as lyapunov_expo
    # Default_dim==> sample_entropy= 2 ,_ ,lyapunov_expo= 10
    
    
    ############ INPUT
    feat_input=[integration, variance, standard_dev,median_absolute_deviation,sample_entropy]   
    embedded_dim=[5] # INPUT
    
    
    # Feature Extraction
    all_feat_list =[]
    
    loop_in=0
    for feat in feat_input:
        
        if (feat=='sample_entropy' or feat=='correlation_dim' or feat=='lyapunov_expo'):
            all_feat_list.append(feature_extraction1(feat,embedded_dim[loop_in]))
            loop_in=loop_in+1
        else:
            all_feat_list.append(feature_extraction(feat))
        
        
    appended_array= np.transpose(np.array(all_feat_list[0]))
    for i in range(1,len(all_feat_list)):
        appended_array =np.concatenate((appended_array,np.transpose(np.array(all_feat_list[i]))),axis=1)

    return appended_array
#######################################
        
file_address0 = r'D:\EEG\EEG dataset\Epilepic Dataset\A'  # INPUT
file_address1 = r'D:\EEG\EEG dataset\Epilepic Dataset\E'  # INPUT
        
# Till now a def file which take two different classes as input
      
#### Clasification Task
X_class0= pd.DataFrame(till_classification(file_address0))

X_class1= pd.DataFrame(till_classification(file_address1))


df = X_class0.append(X_class1)
df= df.reset_index(drop=True)

X= df.iloc[:,:].values     
Y = np.concatenate((np.zeros(shape=len(X_class0)), np.ones(shape=len(X_class1))))


############### Feature Selection
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest

feat_selection_method= 'SelectKBest'
precentage_or_value = 20 # INPUT

if (feat_selection_method== 'SelectPercentile'):
    X = SelectPercentile(percentile=precentage_or_value).fit_transform(X, Y)
elif (feat_selection_method== 'SelectKBest'):
    X = SelectKBest(k=precentage_or_value).fit_transform(X, Y)
else:
    X=X


##############################################

####################
from sklearn.model_selection import KFold, StratifiedKFold

######################
# Chossing k-fold Method

cv_fold =5  # INPUT
k_fold_options= [KFold,StratifiedKFold] # Input in term of Index
k_fold_index=1 # INPUT
#######################

## Chossing Classifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

classifier_array= [SVC,SGDClassifier,KNeighborsClassifier,GaussianProcessClassifier,
                   GaussianNB,DecisionTreeClassifier,AdaBoostClassifier,GradientBoostingClassifier]

input_classifier_index=0 # INPUT

from sklearn.model_selection import StratifiedKFold
skf = k_fold_options[k_fold_index](n_splits=cv_fold,shuffle=True,random_state=59)
accuracy_train_list=[]
accuracy_test_list=[]
sensitivity_list=[]
specificity_list=[]
all_cf_matrix=[]


for train_index, test_index in skf.split(X, Y):
    
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test= sc_X.transform(X_test)


    from sklearn.naive_bayes import GaussianNB    
    rf= classifier_array[input_classifier_index]()
    

    rf.fit(X_train,Y_train)

    
    Y_pred_test=rf.predict(X_test)
    Y_pred_train=rf.predict(X_train)
    
    from sklearn.metrics import roc_auc_score,accuracy_score,confusion_matrix
    
    confusion_matrix=confusion_matrix(Y_test, Y_pred_test)
    all_cf_matrix.append(confusion_matrix)
    
    # Sensitivity, hit rate, recall, or true positive rate
    sensitivity_list.append(round((confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[1][0]))*100,1))
    # Specificity or true negative rate
    specificity_list.append(round((confusion_matrix[1][1]/(confusion_matrix[1][1]+confusion_matrix[0][1]))*100,1))

    accuracy_train_list.append(round((accuracy_score(Y_train,Y_pred_train)*100),1))     
    accuracy_test_list.append(round((accuracy_score(Y_test,Y_pred_test)*100),1)) 

avg_cf_matrix= np.average(np.array(all_cf_matrix),axis=0)
class0_acc= avg_cf_matrix[1][1]/(avg_cf_matrix[1][1]+avg_cf_matrix[1][0])
class1_acc= avg_cf_matrix[0][0]/(avg_cf_matrix[0][0]+avg_cf_matrix[0][1])

sensitivity_avg, sensitivity_std= np.mean(sensitivity_list), np.std(sensitivity_list)
specificity_avg, specificity_std= np.mean(specificity_list), np.std(specificity_list)
accuracy_train_avg, accuracy_train_std= np.mean(accuracy_train_list), np.std(accuracy_train_list)
accuracy_test_avg, accuracy_test_std= np.mean(accuracy_test_list), np.std(accuracy_test_list)


sensitivity_list.append("Avg: {avg}".format(avg =round(sensitivity_avg,1)))
sensitivity_list.append("Std: {std}".format(std =round(sensitivity_std,1)))

specificity_list.append("Avg: {avg}".format(avg =round(specificity_avg,1)))
specificity_list.append("Std: {std}".format(std =round(specificity_std,1)))

accuracy_train_list.append("Avg: {avg}".format(avg =round(accuracy_train_avg,1)))
accuracy_train_list.append("Std: {std}".format(std =round(accuracy_train_std,1)))

accuracy_test_list.append("Avg: {avg}".format(avg =round(accuracy_test_avg,1)))
accuracy_test_list.append("Std: {std}".format(std =round(accuracy_test_std,1)))


stats_dataframe= pd.DataFrame({'Sensitivity':sensitivity_list, 'Specificity':specificity_list,
                               'Train Accuracy':accuracy_train_list , 'Test Accuracy':accuracy_test_list})
boxplot_frame= pd.DataFrame({'Sensitivity':sensitivity_list[0:cv_fold], 'Specificity':specificity_list[0:cv_fold],
                               'Train Accuracy':accuracy_train_list[0:cv_fold] , 'Test Accuracy':accuracy_test_list[0:cv_fold]})

############# Printing Outputs
from IPython.display import display
display(stats_dataframe)

############################ Confusion Matrix Plot
# plt.figure(figsize=(10,8))
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_percentages = ["{0:.2%}".format(value) for value in avg_cf_matrix.flatten()/np.sum(avg_cf_matrix)]
labels = [f"{v1}\n{v2}" for v1, v2 in
          zip(group_names,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(avg_cf_matrix, annot=labels, fmt='', cmap='Blues')
plt.title("confusion_matrix",fontsize=15)
#plt.savefig('plot1.png')
    
########################### Accuracy, Sensitivity and Specificity plot
fig, ax = plt.subplots()
names = ['Specificity','Sensitivity','Accuracy']
x_pos = np.arange(len(names))
CTEs = [specificity_avg, sensitivity_avg, accuracy_test_avg]
error = [specificity_std, sensitivity_std, accuracy_test_std]
# Build the plot
ax.bar(x_pos, CTEs, yerr=error, align='center', capsize=10)
ax.set_xticks(x_pos)
ax.set_xticklabels(names,fontsize = 10)
#ax.set(facecolor = "black")

# show figure
plt.tight_layout()
plt.ylabel("Percentage")
plt.title("Performance",fontsize=15)
#plt.savefig('plot2.png')
plt.show()
    
########################## Classwise Performance
plt.figure()
data={"class":["class 0","class 1"],"accuracy":[class0_acc,class1_acc]}
ax = sns.barplot(x="class", y="accuracy", data=data,edgecolor="white")
ax.set(ylim=(0, 1.2))
#ax.set(facecolor = "black")
ax.text(0, data["accuracy"][0]/2, "{0:.2%}".format(data["accuracy"][0]), ha="center") 
ax.text(1, data["accuracy"][1]/2, "{0:.2%}".format(data["accuracy"][1]), ha="center") 
plt.ylabel("percentage",fontsize = 10)
plt.title("Classwise Performance",fontsize=15)
#plt.savefig('plot3.png') 
    
############################################################ 
    
    
    
    
    
    
    


