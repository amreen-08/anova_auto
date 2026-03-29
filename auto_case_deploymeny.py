import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy import stats

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