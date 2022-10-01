
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############


import streamlit as st
import plotly.express as px

# load iris data
df = px.data.iris()

features = df.columns.to_list()[:-2]
with st.sidebar:
    st.write('2D scatter plot')
    x_feature = st.radio('Horizontal axis',
             features)
    y_feature = st.radio('Vertical axis',
             features)    

#%% original data

with st.expander('Original data'):
    st.write(df)
    

#%% Heatmap
with st.expander('Heatmap'):
    fig_1 = px.imshow(df.iloc[:,0:4],
                      color_continuous_scale='RdYlBu_r')
    st.plotly_chart(fig_1)
    
#%% 2D scatter plot
with st.expander('2D scatter plot'):
    fig_2 = px.scatter(df, x=x_feature, y=y_feature, color="species")
    st.plotly_chart(fig_2)

#%% 3D scatter plot
with st.expander('3D scatter plot'):
    fig_3 = px.scatter_3d(df, 
                          x='sepal_length', 
                          y='sepal_width', 
                          z='petal_width',
                  color='species')
    st.plotly_chart(fig_3)

# Pairwise scatter plot
with st.expander('Pairwise scatter plot'):
    fig_4 = px.scatter_matrix(df, 
                            dimensions=["sepal_width", 
                                        "sepal_length", 
                                        "petal_width", 
                                        "petal_length"], 
                            color="species")
    st.plotly_chart(fig_4)
    
    
    
    
