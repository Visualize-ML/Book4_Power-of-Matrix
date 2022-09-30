
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############


import numpy as np
import streamlit as st
import time

# transition matrix
A = np.matrix([[0.7, 0.2],
               [0.3, 0.8]])


with st.sidebar:
    pi_0_chicken = st.slider('Ratio of chicken:',
              0.0, 1.0, step = 0.1)
    pi_0_rabbit  = 1 - pi_0_chicken
    st.write('Ratio of rabbit: ' + str(round(pi_0_rabbit,1)))
    
    num_iterations = st.slider('Number of nights:',
                               20,100,step = 5)


progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()
last_rows = np.array([[pi_0_chicken, pi_0_rabbit]])
# st.write(last_rows) # row vector

chart = st.line_chart(last_rows)

for i in range(1, num_iterations):
    
    last_status = last_rows[-1,:]
    
    # st.write(last_status)
    
    new_rows = last_status@A.T
    
    percent = (i + 1)*100/num_iterations
    
    status_text.text("%i%% Complete" % percent)
    
    chart.add_rows(new_rows)
    progress_bar.progress(i)
    last_rows = new_rows
    time.sleep(0.1)

progress_bar.empty()

