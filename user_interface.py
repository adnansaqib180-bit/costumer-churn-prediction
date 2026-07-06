import streamlit as st
import joblib 
import pandas as pd

model = joblib.load('logisticmodel.pkl')
scaler = joblib.load('scaler.pkl')