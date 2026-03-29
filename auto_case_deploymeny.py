import streamlit as st
import pandas as pd
import numpy as np
 
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
except ImportError:
    st.error("statsmodels is not installed. Please ensure requirements.txt includes 'statsmodels'.")
    st.stop()
 
try:
    from scipy import stats
except ImportError:
    st.error("scipy is not installed. Please ensure requirements.txt includes 'scipy'.")
    st.stop()
 
# -------------------- UI CONFIG --------------------
st.set_page_config(page_title="ANOVA Analysis - Auto Dataset", layout="wide")
 
st.title("🚗 ANOVA Analysis on Auto Dataset")
st.markdown("Analyze how **cylinders** and **origin** affect MPG")
 
# -------------------- FILE UPLOAD --------------------
uploaded_file = st.file_uploader("Upload Auto.csv", type=["csv"])
 
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
 
    # -------------------- DATA PREPROCESSING --------------------
    df['origin'] = df['origin'].replace({1: 'USA', 2: 'Europe', 3: 'Japan'})
 
    # Convert cylinders to numeric, drop rows where mpg or cylinders is missing
    df['cylinders'] = pd.to_numeric(df['cylinders'], errors='coerce')
    df['mpg'] = pd.to_numeric(df['mpg'], errors='coerce')
    df = df.dropna(subset=['mpg', 'cylinders', 'origin'])
 
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())
 
    # -------------------- MODEL 1: WITHOUT INTERACTION --------------------
    st.subheader("📌 ANOVA (Without Interaction)")
 
    fit1 = ols('mpg ~ cylinders + origin', data=df).fit()
    anova1 = sm.stats.anova_lm(fit1, typ=2)
 
    st.write("### ANOVA Table")
    st.dataframe(anova1)
 
    # -------------------- MODEL 2: WITH INTERACTION --------------------
    st.subheader("📌 ANOVA (With Interaction)")
 
    fit2 = ols('mpg ~ cylinders * origin', data=df).fit()
    anova2 = sm.stats.anova_lm(fit2, typ=2)
 
    st.write("### ANOVA Table")
    st.dataframe(anova2)
 
    # -------------------- INTERPRETATION --------------------
    st.subheader("🧠 Interpretation")
 
    if anova2['PR(>F)']['cylinders:origin'] < 0.05:
        st.success("Interaction between cylinders and origin is SIGNIFICANT")
    else:
        st.warning("No significant interaction between cylinders and origin")
 
    # -------------------- TUKEY TEST --------------------
    st.subheader("📊 Tukey HSD Test")
 
    df['group'] = df['cylinders'].astype(str) + "_" + df['origin']
 
    tukey = pairwise_tukeyhsd(endog=df['mpg'], groups=df['group'])
 
    tukey_df = pd.DataFrame(
        data=tukey._results_table.data[1:],
        columns=tukey._results_table.data[0]
    )
 
    st.write("### Full Tukey Results")
    st.dataframe(tukey_df)
 
    # Significant results
    significant = tukey_df[tukey_df['reject'] == True].copy()
    significant = significant.sort_values(by='meandiff', ascending=False)
 
    st.write("### ✅ Significant Differences (Sorted)")
    st.dataframe(significant)
 
    # -------------------- FILTER FOR MAX DIFFERENCE --------------------
    st.subheader("🔍 Highest Mean Difference")
 
    if not significant.empty:
        top_diff = significant.iloc[0]
        st.write(top_diff)
 
    # -------------------- T-TEST --------------------
    st.subheader("📌 T-Test: Japan (3 vs 4 Cylinders)")
 
    df4 = df[(df['cylinders'] == 3) & (df['origin'] == 'Japan')]
    df5 = df[(df['cylinders'] == 4) & (df['origin'] == 'Japan')]
 
    if df4.empty or df5.empty:
        st.warning("Not enough data for T-Test (3-cylinder or 4-cylinder Japan group is empty).")
    else:
        ttest = stats.ttest_ind(df4['mpg'], df5['mpg'],
                               equal_var=False,
                               alternative='greater')
 
        st.write("### T-Test Result")
        st.write(f"Statistic: {ttest.statistic}")
        st.write(f"P-value: {ttest.pvalue}")
 
        if ttest.pvalue < 0.05:
            st.success("3-cylinder cars in Japan have significantly higher MPG than 4-cylinder")
        else:
            st.warning("No significant difference")
 
else:
    st.info("Please upload the Auto.csv file to proceed.")