# This is a sample Python script.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mp
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import OrdinalEncoder

ord_enc = OrdinalEncoder()

loan_data_backup = pd.read_csv('C:\\Users\\Checkout\\Downloads\\loan_data_2017.csv')
loan_data = loan_data_backup.copy()
# pd.options.display. = None
# print(loan_data.shape)
# print(loan_data.info())
# pd.set_option('display.max_columns', None)
# print(loan_data.head())
# print((loan_data.isnull().sum(axis = 0) / loan_data.shape[0] * 100.00).sort_values(ascending = False).head(50))

loan = loan_data.drop(
    ['member_id', 'orig_projected_additional_accrued_interest', 'hardship_loan_status', 'hardship_dpd',
     'hardship_reason', 'hardship_status', 'deferral_term', 'hardship_amount', 'hardship_start_date',
     'hardship_end_date', 'payment_plan_start_date', 'hardship_length', 'hardship_type',
     'hardship_payoff_balance_amount', 'settlement_percentage', 'debt_settlement_flag_date',
     'settlement_status', 'settlement_date', 'settlement_amount', 'settlement_term',
     'sec_app_mths_since_last_major_derog', 'sec_app_revol_util', 'revol_bal_joint',
     'sec_app_chargeoff_within_12_mths', 'sec_app_num_rev_accts', 'sec_app_open_act_il', 'sec_app_open_acc',
     'sec_app_mort_acc', 'sec_app_inq_last_6mths', 'sec_app_earliest_cr_line',
     'sec_app_collections_12_mths_ex_med', 'verification_status_joint',
     'dti_joint', 'annual_inc_joint', 'desc', 'mths_since_last_record', 'hardship_last_payment_amount'],
    axis=1)

# eliminate all features with more than 80% missing values
# print(loan.shape)

loans = loan.drop(['id', 'zip_code', 'addr_state', 'funded_amnt', 'funded_amnt_inv', 'url'], axis=1)
# print(loans.shape)
# print((loans.isnull().sum(axis = 1) / 106 * 100.00).sort_values(ascending = False).head(20))

# Drop rows with more than 80% missing values
perc = 80.0
min_count = int(((100 - perc) / 100) * loans.shape[1] + 1)
loans = loans.dropna(axis=0, thresh=min_count)
# print(loans.shape)

# check missing value of each column
# print((loans.isnull().sum(axis=0) / loans.shape[0] * 100.00).sort_values(ascending=False).head(50))

# For categorical features: fill in mode
# For numerical features: fill in median
null_mode = ['emp_length', 'title', 'next_pymnt_d', 'emp_title', 'last_pymnt_d', 'last_credit_pull_d',
             'earliest_cr_line']

null_median = ['mths_since_recent_inq', 'num_tl_120dpd_2m', 'mo_sin_old_il_acct', 'bc_util', 'percent_bc_gt_75',
               'bc_open_to_buy', 'mths_since_recent_bc', 'pct_tl_nvr_dlq', 'num_rev_accts', 'mo_sin_old_rev_tl_op',
               'mo_sin_rcnt_rev_tl_op', 'num_actv_bc_tl', 'num_accts_ever_120_pd', 'total_rev_hi_lim', 'num_bc_tl',
               'tot_cur_bal', 'tot_coll_amt', 'num_actv_rev_tl', 'mo_sin_rcnt_tl', 'tot_hi_cred_lim', 'num_op_rev_tl',
               'num_rev_tl_bal_gt_0', 'num_tl_30dpd', 'total_il_high_credit_limit', 'num_tl_90g_dpd_24m',
               'num_tl_op_past_12m', 'num_il_tl', 'num_bc_sats', 'num_sats', 'total_bc_limit', 'acc_open_past_24mths',
               'total_bal_ex_mort', 'mort_acc', 'dti', 'pub_rec_bankruptcies', 'collections_12_mths_ex_med',
               'chargeoff_within_12_mths', 'inq_last_6mths', 'mths_since_recent_bc_dlq', 'mths_since_last_major_derog',
               'mths_since_recent_revol_delinq', 'mths_since_last_delinq', 'il_util', 'mths_since_rcnt_il', 'all_util',
               'open_acc_6m', 'inq_last_12m', 'total_cu_tl', 'open_rv_24m', 'inq_fi', 'max_bal_bc', 'open_act_il',
               'open_rv_12m', 'open_il_24m', 'open_il_12m', 'total_bal_il', 'avg_cur_bal', 'tax_liens', 'total_acc',
               'acc_now_delinq', 'pub_rec', 'open_acc', 'delinq_amnt', 'delinq_2yrs', 'annual_inc']

for feature in null_mode:
    loans[feature] = loans[feature].fillna(loans[feature].mode().iloc[0])

print(loans['acc_now_delinq'])

for feature in null_median:
    loans[feature] = loans[feature].fillna(loans[feature].median())

# Create a column with issue year only without month
loans['issue_year'] = loans['issue_d'].str[-4:]
loans['issue_year'] = loans['issue_year'].astype(int)
print(loans['issue_year'].describe())
# values = (float(x.strip('%')/100) for x in list(loans['int_rate'].values()))

# loans = loans.drop(['issue_d'], axis=1)
# print(loans.head())
val = []
for i in loans['int_rate']:
    val.append(float(i.strip('%')))

print(sns.displot(loans, x="loan_amnt", kde=True, bins=20, color='orange').set(title='Distribution of Loan Amount',
                                                                               xlabel='Loan Amount'))
sns.displot(loans, x="int_rate", kde=True, bins=20).set(title='Distribution of Loan Interest',
                                                        xlabel='Loan Interest')
plt.show()
amount_and_int = loans[['int_rate', 'loan_amnt']]
print(amount_and_int.describe())

# Get average interest rate of loans by grade
# print("Group By: ", loans.groupby(['grade', 'int_rate'], as_index=False).mean(numeric_only=True))

# loans.groupby('grade')['int_rate'].mean()
loans['int_rate'] = list(map(float, val))


interest_group = loans.groupby('grade')['int_rate'].mean()

# interest_group = pd.DataFrame(interest_group , c='grade')
interest_group = pd.DataFrame(interest_group)
interest_group['grade'] = interest_group.index
interest_group.rename(columns={'int_rate': 'ave_int_rate'}, inplace=True)
interest_group.reset_index(inplace=True, drop=True)
print(interest_group)

grade_int = loans[['grade', 'int_rate']]
grade_int = grade_int.sort_values('grade', ascending=True)
sns.boxplot(x='grade', y="int_rate", data=grade_int, palette="Pastel2").set(title='Loan Interest Rate Range by Grade', xlabel='Grade',
                                                         ylabel='Loan Interest Rate')
fig = plt.gcf()
plt.show()

print(grade_int.groupby(['grade'])['int_rate'].describe())
grade_amount = loans[['grade', 'loan_amnt']]
grade_amount = grade_amount.sort_values('grade', ascending=True)
sns.boxplot(x="grade", y="loan_amnt", data=grade_amount, palette="Pastel2").set(title='Loan Amount Range of Each Grade', xlabel='Grade',
                                                             ylabel='Loan Amount in US$')
sns.despine(offset=10, trim=True)

fig = plt.gcf()
plt.show()

grade_amount.groupby(['grade'])['loan_amnt'].describe()
amount_grade = loans.groupby(['grade'])['loan_amnt'].sum()
amount_grade = pd.DataFrame(amount_grade)
amount_grade['grade'] = amount_grade.index
# amount_grade.rename(columns={'int_rate':'ave_int_rate'}, inplace=True)
amount_grade.reset_index(inplace=True, drop=True)
print(amount_grade)

sns.barplot(data=amount_grade, x='grade', y='loan_amnt').set(title='Loan Amount by Grade', xlabel='Grade',
                                                             ylabel='Loan Amount in US$')
fig = plt.gcf()
plt.show()

# Get loan amount by year
print(loans['issue_year'])
year_amount = loans.groupby(['issue_year'])['loan_amnt'].sum()
year_amount = pd.DataFrame(year_amount)
year_amount['issue_year'] = year_amount.index
year_amount.reset_index(inplace=True, drop=True)
year_amount = year_amount.sort_values('issue_year', ascending=True)
print(year_amount)
sns.lineplot(x='issue_year', y="loan_amnt", data=year_amount).set(title='Loan Amount by Year', xlabel='Year',
                                                                  ylabel='Loan Amount in US$')
sns.despine(offset=10, trim=True)
fig = plt.gcf()
year_grade = loans.groupby(['issue_year', 'grade'])['loan_amnt'].sum()
year_grade = pd.DataFrame(year_grade)
year_grade['info'] = year_grade.index.to_numpy()
year_grade['year'] = year_grade['info'].str[0]
year_grade['grade'] = year_grade['info'].str[1]
year_grade.reset_index(inplace=True, drop=True)
year_grade = year_grade.drop(['info'], axis=1)
print(year_grade)

sns.lineplot(x='year', y="loan_amnt", palette='Pastel2', data=year_grade).set(title='Loan Amount by Grade', xlabel='Year',
                                                                        ylabel='Loan Amount in US$')
sns.despine(offset=10, trim=True)
# fig = plt.gcf()
plt.show()
term_interest = loans.groupby(['term'])['int_rate'].mean()
term_interest = pd.DataFrame(term_interest)
term_interest['term'] = term_interest.index
term_interest.reset_index(inplace=True, drop=True)
print(term_interest)

term_amount = loans.groupby(['term'])['loan_amnt'].sum()
term_amount = pd.DataFrame(term_amount)
term_amount['term'] = term_amount.index
term_amount.reset_index(inplace=True, drop=True)
print(term_amount)

pie, ax = plt.subplots(figsize=[10, 6])
labels = term_amount['term']
colors = sns.color_palette('pastel')[0:2]

plt.pie(x=term_amount['loan_amnt'], autopct="%.1f%%", labels=labels, explode=[0.05, 0], pctdistance=0.5, colors= colors)
plt.title('Percentage of Loan Amount by Term', fontsize=14)
plt.show()

term_amount_grade = loans.groupby(['grade', 'term'])['loan_amnt'].sum()
term_amount_grade = pd.DataFrame(term_amount_grade)
term_amount_grade['info'] = term_amount_grade.index.to_numpy()
term_amount_grade['grade'] = term_amount_grade['info'].str[0]
term_amount_grade['term'] = term_amount_grade['info'].str[1]
term_amount_grade.reset_index(inplace=True, drop=True)
term_amount_grade = term_amount_grade.drop(['info'], axis=1)
print(term_amount_grade)


sns.barplot(x='grade', y='loan_amnt', palette='Blues', data=term_amount_grade).set(
    title='Loan Amount by Grade and Term', xlabel='Grade', ylabel='Loan Amount in US$')
fig = plt.gcf()
fig.set_size_inches(9, 6)
plt.show()

print("Loan Status: ")
print(loans['loan_status'].value_counts())

loans['loan_status'].replace({'Does not meet the credit policy. Status:Fully Paid': 'Fully Paid',
                              'Does not meet the credit policy. Status:Charged Off': 'Charged Off'}, inplace=True)
status = loans['loan_status'].value_counts()
status = pd.DataFrame(status)
status = status.rename(columns={'loan_status': '#_of_loans'})
status['loan_status'] = status.index
status.reset_index(inplace=True, drop=True)
print(status)

pie, ax = plt.subplots(figsize=[10, 6])
label = status['loan_status']
plt.pie(x=status['#_of_loans'], autopct="%.1f%%", labels=label, pctdistance=0.5, colors=colors)
plt.legend(label)
plt.title('Number of Loans by Status', fontsize=14)
plt.show()

print("Number of loans that are fully paid or defaulted: ")
loans['loan_status'].replace({'Charged Off': 'Default'}, inplace=True)
status2 = loans['loan_status'].value_counts()
status2 = pd.DataFrame(status2)
status2 = status2.rename(columns={'loan_status': '#_of_loans'})
status2['loan_status'] = status2.index
status2.reset_index(inplace=True, drop=True)
status2 = status2.drop([0, 3, 4, 5])
print(status2)

pie, ax = plt.subplots(figsize=[10, 6])
label2 = status2['loan_status']
plt.pie(x=status2['#_of_loans'], autopct="%.1f%%", labels=label2, pctdistance=0.5, colors= colors)
plt.legend(label2)
plt.title('Number of Loans by Status', fontsize=14)
plt.show()

sns.boxplot(x='home_ownership', y="int_rate", data=loans , palette="Pastel2").set(title='Interest Rate by Home Ownership',
                                                              xlabel='Home Ownership',
                                                              ylabel='Interest Rate')
sns.despine(offset=10, trim=True)
fig = plt.gcf()
plt.show()

home_owner = loans[['home_ownership', 'int_rate', 'loan_amnt']]
home_owner['home_ownership_code'] = ord_enc.fit_transform(home_owner[['home_ownership']])
mortage = home_owner.drop([1.0, 2.0])
print(mortage.head())

mask1 = np.triu(np.ones_like(home_owner.corr(), dtype=np.bool))
heatmap1 = sns.heatmap(home_owner.corr(), mask=mask1, vmin=-1, vmax=1, annot=True, cmap="Blues")
heatmap1.set_title('Correlation Heatmap')
plt.show()

purpose = loans.groupby(['purpose'])['loan_amnt'].sum()
purpose = pd.DataFrame(purpose)
purpose['purpose'] = purpose.index
purpose = purpose.sort_values('loan_amnt', ascending=False)
purpose.reset_index(inplace=True, drop=True)
print(purpose)

sns.barplot(x='loan_amnt', y='purpose', data=purpose ,  palette="Blues").set(title='Loan Amount by Purpose', xlabel='Loan Amount',
                                                          ylabel='Purpose')
sns.despine(offset=10, trim=True)
fig = plt.gcf()
fig.set_size_inches(14, 6)
plt.show()

purpose = loans[['purpose', 'int_rate', 'loan_amnt']]
purpose['purpose_code'] = ord_enc.fit_transform(purpose[['purpose']])

mask2 = np.triu(np.ones_like(purpose.corr(), dtype=np.bool))
heatmap2 = sns.heatmap(purpose.corr(), mask=mask2, vmin=-1, vmax=1, annot=True, cmap="Blues")
heatmap2.set_title('Correlation Heatmap')
plt.show()

print("Prepare dataset for machine learning")

# select samples which are either fully paid or default
loans1 = loans[loans['loan_status'].str.contains('Fully Paid')]
loans2 = loans[loans['loan_status'].str.contains('Default')]

# combine fully paid and default samples
# set fully paid to 1 and default to 0
df = pd.concat([loans1, loans2], ignore_index=True)
df['loan_status'].replace({'Fully Paid': 1, 'Default': 0}, inplace=True)

# issue_year is created. Therefore drop issue_d
# pymnt_plan contains only one single value
# hardship_flag contains only one single value
df = df.drop(['issue_d', 'pymnt_plan', 'hardship_flag'], axis=1)
# print(df.shape)
#
# print(df.head())

var = df.select_dtypes(exclude="object")
vif_data = pd.DataFrame()
vif_data["feature"] = var.columns
vif_data["VIF"] = [variance_inflation_factor(var.values, i) for i in range(len(var.columns))]
print(vif_data[vif_data["VIF"] > 10].feature)

high_vif = ['loan_amnt', 'installment', 'open_acc', 'total_acc', 'out_prncp',
            'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',
            'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'policy_code', 'tot_cur_bal',
            'num_actv_rev_tl',
            'num_il_tl', 'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats',
            'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit']

# convert categorical features to numerical for models other than linear regressions
df['term_code'] = ord_enc.fit_transform(df[['term']])
df['grade_code'] = ord_enc.fit_transform(df[['grade']])
df['sub_grade_code'] = ord_enc.fit_transform(df[['sub_grade']])
df['emp_title_code'] = ord_enc.fit_transform(df[['emp_title']])
df['home_ownership_code'] = ord_enc.fit_transform(df[['home_ownership']])
df['verification_status_code'] = ord_enc.fit_transform(df[['verification_status']])
df['purpose_code'] = ord_enc.fit_transform(df[['purpose']])
df['title_code'] = ord_enc.fit_transform(df[['title']])
df['earliest_cr_line_code'] = ord_enc.fit_transform(df[['earliest_cr_line']])
df['initial_list_status_code'] = ord_enc.fit_transform(df[['initial_list_status']])
df['last_pymnt_d_code'] = ord_enc.fit_transform(df[['last_pymnt_d']])
df['next_pymnt_d_code'] = ord_enc.fit_transform(df[['next_pymnt_d']])
df['last_credit_pull_d_code'] = ord_enc.fit_transform(df[['last_credit_pull_d']])
df['application_type_code'] = ord_enc.fit_transform(df[['application_type']])
df['disbursement_method_code'] = ord_enc.fit_transform(df[['disbursement_method']])
df['debt_settlement_flag_code'] = ord_enc.fit_transform(df[['debt_settlement_flag']])
df['emp_length_code'] = ord_enc.fit_transform(df[['emp_length']])
df['revol_util'] = ord_enc.fit_transform(df[['revol_util']])

# drop the original categorical features

df = df.drop(['term', 'grade', 'sub_grade', 'emp_title', 'home_ownership', 'verification_status', 'purpose',
              'title', 'earliest_cr_line', 'initial_list_status', 'last_pymnt_d', 'next_pymnt_d',
              'last_credit_pull_d', 'application_type', 'disbursement_method', 'debt_settlement_flag',
              'emp_length'], axis=1)
print(df.shape)
print(df.head())
df.to_csv('data_non_linear.csv')
df.drop(high_vif, axis=1, inplace=True)
df.shape
print(df.head())

df.to_csv('data_linear.csv')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
