#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import plotly.graph_objs as go
import numpy as np

# Function to calculate the gradient descent steps
def gradient_descent(learning_rate, iterations, start_x, start_y):
    x_vals = [start_x]
    y_vals = [start_y]
    for _ in range(iterations):
        grad_x = 2 * x_vals[-1]
        grad_y = 2 * y_vals[-1]
        new_x = x_vals[-1] - learning_rate * grad_x
        new_y = y_vals[-1] - learning_rate * grad_y
        x_vals.append(new_x)
        y_vals.append(new_y)
    return x_vals, y_vals

# App layout
st.title("Introduction to Gradient Descent")

st.write("""
Gradient Descent is a fundamental optimization algorithm used in machine learning to minimize a function by iteratively moving towards the minimum. It's inspired by how nature moves through complex environments, gradually finding a path that reduces error.
""")

st.subheader("How It Works:")
st.markdown("""
1. **The Mountain and Fog Example:** Imagine standing on a mountain surrounded by thick fog. You can't see anything, but you can feel the ground under your feet. To find the lowest point, you follow the slope, moving carefully downhill with each step.
2. **The Water Flow Example:** Think of how water flows downhill, naturally following the steepest path. Similarly, Gradient Descent follows the slope of the function to find the minimum value, adjusting the model parameters to minimize error.
""")

# Sidebar for user inputs
st.sidebar.header("Parameters")
learning_rate = st.sidebar.slider("Learning Rate:", 0.01, 1.0, 0.1, 0.01)
iterations = st.sidebar.number_input("Number of Iterations:", min_value=1, value=100)
start_x = st.sidebar.slider("Starting Point X:", -10.0, 10.0, 9.0)
start_y = st.sidebar.slider("Starting Point Y:", -10.0, 10.0, -10.0)

# Calculate gradient descent path
x_vals, y_vals = gradient_descent(learning_rate, iterations, start_x, start_y)

# Create the 3D plot using Plotly
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

fig = go.Figure()
fig.add_trace(go.Surface(z=Z, x=X, y=Y, opacity=0.7, colorscale='Viridis'))
fig.add_trace(go.Scatter3d(x=x_vals, y=y_vals, z=[x**2 + y**2 for x, y in zip(x_vals, y_vals)], 
                           mode='lines+markers', name='Gradient Descent', marker=dict(size=5, color='red')))

fig.update_layout(scene=dict(
    xaxis=dict(title='X'),
    yaxis=dict(title='Y'),
    zaxis=dict(title='Loss (f(x, y))')
))

# Display the plot
st.plotly_chart(fig)

