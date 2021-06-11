import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def draw_meshgrid():
    a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
    b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)

    XX, YY = np.meshgrid(a, b)

    input_array = np.array([XX.ravel(), YY.ravel()]).T

    return XX, YY, input_array

X,y=make_moons(n_samples=300,noise=0.2,random_state=20)
X_train,X_test,y_train,y_test=train_test_split(X,y)


st.sidebar.markdown("## eXtreme Gradient Boosting Dashboard")

# dataset=st.sidebar.selectbox(
#     'Dataset',
#     ('DS1','DS2')
# )

tuning=st.sidebar.radio(
    'Hyper Parameter',
    ('Default Value','Tuning')
)
if tuning=='Tuning':
    st.sidebar.info('Initially, all parameter values are set to default. Change them according to your need.:smiley:')

    base_score=st.sidebar.number_input('Base Score',min_value=0.01,max_value=0.99,format='%.2f',value=0.5)

    booster=st.sidebar.selectbox(
        'Booster',
        ('gbtree','gblinear','dart')
    )

    if booster=='gbtree' or booster=='dart':
        colsample_bylevel=st.sidebar.number_input('Column Sample by Level',min_value=0.00,max_value=1.00,value=1.00)

        colsample_bynode=st.sidebar.number_input('Column Sample by Node',min_value=0.00,max_value=1.00,value=1.00)

        colsample_bytree=st.sidebar.number_input('Column Sample by Tree',min_value=0.00,max_value=1.00,value=1.00)

        gamma=st.sidebar.number_input('Gamma',min_value=0)

        gpu_id = st.sidebar.number_input('GPU Id', value=-1, min_value=-1)

        learning_rate=st.sidebar.number_input('Learning Rate',min_value=0.00,max_value=1.00,value=0.3)

        max_delta_step = st.sidebar.number_input('Maximum Delta Step', min_value=0)

        max_depth=st.sidebar.number_input('Maximum Depth',min_value=0,value=6)

        min_child_weight=st.sidebar.number_input('Minimum Child Weight',min_value=0,value=1)

    else:
        colsample_bylevel = 1
        colsample_bynode = 1
        colsample_bytree = 1
        gamma = 0
        learning_rate=0.3
        gpu_id=-1
        max_delta_step=0
        max_depth=6
        min_child_weight=1

    clf=XGBClassifier(base_score=base_score,booster=booster,colsample_bylevel=colsample_bylevel,colsample_bynode=colsample_bynode,
                      colsample_bytree=colsample_bytree,gamma=gamma,gpu_id=gpu_id,learning_rate=learning_rate,
                      max_delta_step=max_delta_step,max_depth=max_depth,min_child_weight=min_child_weight)
    # clf = GradientBoostingClassifier(loss, learning_rate, n_estimators, subsample, criterion, min_samples_split,
    #                                  min_samples_leaf, min_weight_fraction_leaf, max_depth ,min_impurity_decrease,
    #                                  min_impurity_split, init, random_state, max_features, verbose, max_leaf_nodes,
    #                                  warm_start, validation_fraction,n_iter_no_change,tol,ccp_alpha)
else:
    clf=XGBClassifier()


fig,ax=plt.subplots()

ax.scatter(X.T[0],X.T[1],c=y,cmap='rainbow')
orig=st.pyplot(fig)


if st.sidebar.button('Run Algorithm'):
    with st.spinner('Your model is getting trained..:muscle:'):
        orig.empty()

        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)

        XX,YY,input_array=draw_meshgrid()
        labels=clf.predict(input_array)

        ax.contourf(XX,YY,labels.reshape(XX.shape),alpha=0.3,cmap='rainbow')

        plt.xlabel('Col1')
        plt.ylabel('Col2')
        orig=st.pyplot(fig)
        st.sidebar.subheader("Accuracy of the model: "+str(round(accuracy_score(y_test,y_pred),2)))
    st.success("Done!")
# subprocess = subprocess.Popen(clf,shell=True, stdout=subprocess.PIPE)
# subprocess_return = subprocess.stdout.read()
# print(subprocess_return)