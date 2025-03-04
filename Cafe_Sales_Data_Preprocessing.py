import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import warnings


warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

def load_data():
    data = pd.read_csv("datasets/dirty_cafe_sales.csv")
    return data
def check_data(dataframe):
    print("########################## HEAD ##########################")
    print(dataframe.head())
    print("########################## INFO ##########################")
    print(dataframe.info())
    print("########################## SHAPE ##########################")
    print(dataframe.shape)
    print("########################## ISNULL(?) ##########################")
    print(dataframe.isnull().sum())
    print("########################## DESCRIBE ##########################")
    print(dataframe.describe().T)
    print("####################################################")

df = load_data()
check_data(df)

# Değişken tiplerinin dönüşümü
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["Price Per Unit"] = pd.to_numeric(df["Price Per Unit"], errors="coerce")
df["Total Spent"] = pd.to_numeric(df["Total Spent"], errors="coerce")

df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce")

df['Item'] = df['Item'].astype('category')
df['Payment Method'] = df['Payment Method'].astype('category')
df['Location'] = df['Location'].astype('category')

# Boş değerleri tespit etmek , veri setinde UNKNOWN ve ERROR olan yerleri NAN ile doldurmak
df.replace(["ERROR", "UNKNOWN"], np.nan, inplace=True)
# df["Payment Method"] = df["Payment Method"].replace(["ERROR", "UNKNOWN"], np.nan)
df.isnull().sum()

# Eksik değerleri doldurmak
# Burada Total spent , unit price, ve quantity değişkenlerini arasındaki metamatik ile eksik değerleri gidderdik.
def fill_missing_values(df):
    """
    Quantity, Price Per Unit ve Total Spent değişkenlerindeki eksik değerleri dolduran fonksiyon.

    - Total Spent, eğer Quantity ve Price Per Unit doluysa çarpılarak hesaplanır.
    - Price Per Unit, eğer Total Spent ve Quantity doluysa hesaplanır.
    - Quantity, eğer Total Spent ve Price Per Unit doluysa hesaplanır.
    """
    df["Total Spent"] = df["Total Spent"].fillna(
        (df["Quantity"] * df["Price Per Unit"]).where(df["Quantity"].notna() & df["Price Per Unit"].notna())
    )

    df["Price Per Unit"] = df["Price Per Unit"].fillna(
        (df["Total Spent"] / df["Quantity"]).where(df["Total Spent"].notna() & df["Quantity"].notna())
    )

    df["Quantity"] = df["Quantity"].fillna(
        (df["Total Spent"] / df["Price Per Unit"]).where(df["Total Spent"].notna() & df["Price Per Unit"].notna())
    )

    return df


##############################################
df.groupby("Item").agg({"Price Per Unit" :"mean",
                        "Quantity": "mean"})

df.groupby("Item").agg({"Price Per Unit" :"mean"})

# buurada price per unit değişkenini ıtem daki ortalama ile doldurduk.
# bu kodun çalışma mantığı
#df.loc[(df["Price Per Unit"].isnull()) & (df["Item"] == "Salad"), "Item"] = df.groupby("Item")["Price Per Unit"].mean()["Salad"]
df.loc[df["Price Per Unit"].isnull(), "Price Per Unit"] = df.groupby("Item")["Price Per Unit"].transform("mean")

# Daha sonra quantity değişkeni için tekrar bu kodu çalıştırıyoruz ki price per unitten kaynaklanan eksiklikler doldurulsun diye
df = fill_missing_values(df)


# ITEm değişkeni Üzzerinde işlemler
df["Item"].value_counts()
df.drop(df[df["Item"]==5.000].index, axis=0, inplace=True)
df.groupby("Item").agg({"Price Per Unit" :"mean"})
# Fiyatları ve karşılık gelen Item değerlerini bir sözlükte tanımla
price_to_item = {
    3.000: "Juice",
    2.000: "Coffee",
    1.000: "Cookie",
    3.000: "Cake",
    5.000: "Salad",
    4.000: "Sandwich",
    4.000: "Smoothie",
    1.500: "Tea"
}

# Eksik Item'ları, Price Per Unit'e göre doldur
df.loc[df["Item"].isnull(), "Item"] = df["Price Per Unit"].map(price_to_item)
df["Item"].value_counts()
# Quantity değişkenini Item değişkeni kırılımında ortalamalar ile doldurdum.
df.loc[df["Quantity"].isnull(), "Quantity"] = df.groupby("Item")["Quantity"].transform("mean")
df = fill_missing_values(df) # Eksik olan değerleri yine doldurduk
#Price per uunit teki eksik değerleri ortalama ile doldurkduk
df.loc[df["Price Per Unit"].isnull(), "Price Per Unit"] = df["Price Per Unit"].mean()
df.loc[df["Item"].isnull(), "Item"] = df["Item"].mode()[0] # Item daki eksikleri mode ile doldurduk.

df.isnull().sum()

###########################################################
### Payment Method
print(f"Null Values:{df["Payment Method"].isna().sum()}\n\n{df["Payment Method"].value_counts()}")

# Ödeme yöntemlerinin veri setindeki oranlarını hesapla
payment_probs = df["Payment Method"].value_counts(normalize=True)


# Boş olan yerlere bu dağılıma uygun rastgele değerler ata
df.loc[df["Payment Method"].isna(), "Payment Method"] = np.random.choice(
    payment_probs.index,  # Ödeme yöntemleri ["Digital Wallet", "Credit Card", "Cash"]
    size=df["Payment Method"].isna().sum(),  # Eksik olan satır sayısı kadar seçim yap
    p=payment_probs.values) # Seçimlerin olasılıklarını belirle

df.isnull().sum()

###########################################################
### Location

print(f"Null Values:{df["Location"].isna().sum()}\n\n{df["Location"].value_counts()}")

# Ödeme yöntemlerinin veri setindeki oranlarını hesapla
location_probs = df["Location"].value_counts(normalize=True)


# Boş olan yerlere bu dağılıma uygun rastgele değerler ata
df.loc[df["Location"].isna(), "Location"] = np.random.choice(
    location_probs.index,  # Ödeme yöntemleri ["Digital Wallet", "Credit Card", "Cash"]
    size=df["Location"].isna().sum(),  # Eksik olan satır sayısı kadar seçim yap
    p=location_probs.values) # Seçimlerin olasılıklarını belirle

df.isnull().sum()


###########################################################
### Transaction Date

df["Transaction Date"].value_counts().head()

transaction_date_probs = df["Transaction Date"].value_counts(normalize=True).head(20)

# Olasılıkların toplamının 1 olduğundan emin olalım
transaction_date_probs = transaction_date_probs / transaction_date_probs.sum()

df.loc[df["Transaction Date"].isna(), "Transaction Date"] = np.random.choice(
    transaction_date_probs.index,  # Ödeme yöntemleri ["Digital Wallet", "Credit Card", "Cash"]
    size=df["Transaction Date"].isna().sum(),  # Eksik olan satır sayısı kadar seçim yap
    p=transaction_date_probs.values) # Seçimlerin olasılıklarını belirle




df["Transaction Date"] = df["Transaction Date"].ffill()  # Önceki değeri kullan

df["Transaction Date"] = df["Transaction Date"].bfill()  # Sonraki değeri kullan

df.isna().sum()


#####################################################################
### Outliers

# Set the figure style
sns.set(style="whitegrid")

# Creating figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(20, 18))

sns.boxplot(x=df["Quantity"], ax=axes[0, 0])
axes[0, 0].set_title("Boxplot for Quantity")

sns.boxplot(x=df["Price Per Unit"], ax=axes[0, 1])
axes[0, 1].set_title("Boxplot for Price Per Unit")

sns.boxplot(x=df["Total Spent"], ax=axes[1, 0])
axes[1, 0].set_title("Boxplot for Total Spent")

plt.show()

#######################################################################
# Visualization

# Set the figure style
sns.set(style="darkgrid")

# Creating figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(18, 15))

# Plot 1: Quantity vs Price Per Unit
sns.scatterplot(x=df['Quantity'], y=df['Price Per Unit'], alpha=0.5, ax=axes[0, 0])
axes[0, 0].set_title("Quantity vs Price Per Unit")

# Plot 2: Location vs Total Spent
sns.boxplot(x=df['Location'], y=df['Total Spent'], ax=axes[0, 1])
axes[0, 1].set_title("Location vs Total Spent")
axes[0, 1].tick_params(axis='x', rotation=45)

# Plot 3: Payment Method vs Total Spent
sns.boxplot(x=df['Payment Method'], y=df['Total Spent'], ax=axes[1, 0])
axes[1, 0].set_title("Payment Method vs Total Spent")
axes[1, 0].tick_params(axis='x', rotation=45)

# Plot 4: Heatmap of Correlation Matrix
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=axes[1, 1])
axes[1, 1].set_title("Correlation Heatmap")

# Adjusting layout and showing plots
plt.tight_layout()
plt.show()