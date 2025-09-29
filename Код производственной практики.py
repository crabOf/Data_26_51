import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scipy.stats as stats
from scipy.stats import shapiro, norm, probplot, jarque_bera,pearsonr,spearmanr
from statsmodels.stats.diagnostic import lilliefors, linear_reset
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

df = pd.read_csv('transformed_companies_cleaned.csv') 

pd.set_option('display.float_format', '{:.2f}'.format)

print("Информация о данных:")
print(f"Размер датасета: {df.shape}")
print(f"Количество строк: {df.shape[0]}")
print(f"Количество столбцов: {df.shape[1]}")

# Анализ пропусков
print("\nАнализ пропущенных значений:")
missing_data = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100
missing_info = pd.DataFrame({
    'Количество пропусков': missing_data,
    'Процент пропусков': missing_percent
})
print(missing_info[missing_info['Количество пропусков'] > 0])

def fill_missing_values_by_group(df):
    # Создаем копию данных, чтобы не изменять исходный DataFrame
    df_filled = df.copy()
    
    # Создаем группы по выручке (если выручка доступна)
    if 'revenue' in df_filled.columns and df_filled['revenue'].notnull().sum() > 5:
        try:
            # Создаем 5 групп по выручке
            df_filled['revenue_group'] = pd.qcut(df_filled['revenue'], q=5, labels=False, duplicates='drop')
            print("Создано 5 групп по выручке")
        except:
            # Если не получается создать 5 групп (например, мало данных), создаем меньше групп
            df_filled['revenue_group'] = pd.qcut(df_filled['revenue'], q=3, labels=False, duplicates='drop')
            print("Создано 3 группы по выручке (недостаточно данных для 5 групп)")
    else:
        # Если выручка отсутствует или мало данных, создаем группы по уставному капиталу
        if 'authorized_capital' in df_filled.columns and df_filled['authorized_capital'].notnull().sum() > 5:
            try:
                df_filled['revenue_group'] = pd.qcut(df_filled['authorized_capital'], q=5, labels=False, duplicates='drop')
                print("Создано 5 групп по уставному капиталу (выручка недоступна)")
            except:
                df_filled['revenue_group'] = pd.qcut(df_filled['authorized_capital'], q=3, labels=False, duplicates='drop')
                print("Создано 3 группы по уставному капиталу (недостаточно данных для 5 групп)")
        else:
            # Если ни выручка, ни уставный капитал недоступны, создаем одну группу (все компании)
            df_filled['revenue_group'] = 0
            print("Все компании объединены в одну группу (недостаточно данных для сегментации)")

    # Определяем колонки для заполнения
    # Исключаем: company_id, company_name, city, registration_date, revenue_group
    exclude_columns = ['company_id', 'company_name', 'city', 'registration_date', 'revenue_group', 'revenue']
    numeric_columns = df_filled.select_dtypes(include=[np.number]).columns.tolist()
    columns_to_fill = [col for col in numeric_columns if col not in exclude_columns]
    
    # Статистика по заполнению
    total_filled = 0
    filled_per_column = {}
    
    # Заполняем пропуски для каждой колонки
    for col in columns_to_fill:
        initial_missing = df_filled[col].isna().sum()
        if initial_missing == 0:
            continue
            
        filled_count = 0
        for group in df_filled['revenue_group'].unique():
            # Создаем маску для текущей группы
            mask = (df_filled['revenue_group'] == group) & df_filled[col].isna()
            
            # Вычисляем медиану для текущей группы
            group_median = df_filled[df_filled['revenue_group'] == group][col].median()
            
            # Заполняем пропуски медианой
            if not np.isnan(group_median):
                df_filled.loc[mask, col] = group_median
                filled_count += mask.sum()
        
        # Сохраняем статистику
        filled_per_column[col] = filled_count
        total_filled += filled_count
        
    # Удаляем временный столбец revenue_group
    if 'revenue_group' in df_filled.columns:
        df_filled = df_filled.drop('revenue_group', axis=1)
    
    print(f"\nВсего заполнено {total_filled} пропусков в {len(filled_per_column)} колонках")
    
    # Проверяем оставшиеся пропуски
    remaining_missing = df_filled.isna().sum().sum()
    if remaining_missing > 0:
        print(f"Осталось {remaining_missing} пропусков, которые не удалось заполнить")
        # Для оставшихся пропусков используем общую медиану
        for col in columns_to_fill:
            if df_filled[col].isna().sum() > 0:
                overall_median = df_filled[col].median()
                if not np.isnan(overall_median):
                    initial_missing = df_filled[col].isna().sum()
                    df_filled[col] = df_filled[col].fillna(overall_median)
                    filled_count = initial_missing - df_filled[col].isna().sum()
                    print(f"  - Дополнительно заполнено {filled_count} пропусков в '{col}' общей медианой")
    
    return df_filled

df = fill_missing_values_by_group(df)

# Выделим числовые колонки для анализа
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
# Уберем идентификаторы из анализа
exclude_columns = ['company_id', 'inn', 'property_form_numeric', 'company_type_numeric']
numeric_columns = [col for col in numeric_columns if col not in exclude_columns]

def PC(df):
    
    # Проверка типа компании (должен быть от 1 до 8)
    if 'company_type_numeric' in df.columns:
        недопустимые = df[~df['company_type_numeric'].between(1, 8)]
        if not недопустимые.empty:
            print(f"Ошибка: найдено {len(недопустимые)} недопустимых значений в 'company_type_numeric'. Допустимые значения: 1-8")
    
    # Проверка формы собственности (должен быть от 1 до 11)
    if 'property_form_numeric' in df.columns:
        недопустимые = df[~df['property_form_numeric'].between(1, 11)]
        if not недопустимые.empty:
            print(f"Ошибка: найдено {len(недопустимые)} недопустимых значений в 'property_form_numeric'. Допустимые значения: 1-11")
    
# Проверяем данные
PC(df)

# Если нужно исправить недопустимые значения
df['company_type_numeric'] = df['company_type_numeric'].clip(1, 8)
df['property_form_numeric'] = df['property_form_numeric'].clip(1, 11)

def three_sigma_rule_outliers(df, numeric_columns):
    outliers_3sigma = {}
    
    for col in numeric_columns:
        if df[col].notna().sum() > 0:  # Проверяем, что есть не-NaN значения
            mean = df[col].mean()
            std = df[col].std()
            
            if std > 0:  # Избегаем деления на ноль
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                outlier_count = len(outliers)
                
                if outlier_count > 0:
                    outliers_3sigma[col] = {
                        'count': outlier_count,
                        'mean': mean,
                        'std': std,
                        'outliers': outliers.tolist()
                    }
    
    return outliers_3sigma

sigma_outliers = three_sigma_rule_outliers(df, numeric_columns)
if sigma_outliers:
    print("Выбросы по правилу 3 сигм:")
    for col, info in sigma_outliers.items():
        print(f"  {col}: {info['count']} выбросов")
else:
    print("Выбросы по правилу 3 сигм не обнаружены")

def boxplot_outliers(df, numeric_columns):
    """Анализ выбросов с помощью ящичковых диаграмм"""
    boxplot_outliers_info = {}
    
    for col in numeric_columns:
        if df[col].notna().sum() > 0:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            outlier_count = len(outliers)
            
            if outlier_count > 0:
                boxplot_outliers_info[col] = {
                    'count': outlier_count,
                    'Q1': Q1,
                    'Q3': Q3,
                    'IQR': IQR,
                    'outliers': outliers.tolist()
                }
    
    return boxplot_outliers_info

boxplot_outliers_info = boxplot_outliers(df, numeric_columns)
if boxplot_outliers_info:
    print("Выбросы по методу ящичковых диаграмм:")
    for col, info in boxplot_outliers_info.items():
        print(f"  {col}: {info['count']} выбросов")
else:
    print("Выбросы по методу ящичковых диаграмм не обнаружены")


# Создаем графики для основных числовых переменных
main_numeric_cols = ['revenue', 'net_profit', 'authorized_capital', 'contracts_count', 
                    'customer_sum', 'supplier_sum', 'inspections_count', 'inspections_violations'
                    'inspections_knm', 'plaintiff_cases', 'defendant_cases', 
                     'licenses_count', 'trademarks_count'
                    ]

fig, axes = plt.subplots(4, 4, figsize=(20, 20))
axes = axes.ravel()

for i, col in enumerate(main_numeric_cols[:]):
    if col in df.columns and df[col].notna().sum() > 0:
        df.boxplot(column=col, ax=axes[i])
        axes[i].set_title(f'Выбросы: {col}')
        axes[i].tick_params(axis='x', rotation=45)

# Убираем лишние subplots
for i in range(len(main_numeric_cols), 16):
    axes[i].set_visible(False)

plt.tight_layout()
plt.show()

def Del(df):
    
    initial_count = len(df)
    rows_to_drop = set()
    
    # Удаляем строки с недопустимыми значениями для company_type_numeric
    if 'company_type_numeric' in df.columns:
        invalid_mask = ~df['company_type_numeric'].between(1, 8)
        invalid_count = invalid_mask.sum()
        if invalid_count > 0:
            rows_to_drop.update(df[invalid_mask].index.tolist())
            print(f"Удалено {invalid_count} строк с недопустимыми значениями в 'company_type_numeric' (должно быть от 1 до 8)")
    
    # Удаляем строки с недопустимыми значениями для property_form_numeric
    if 'property_form_numeric' in df.columns:
        invalid_mask = ~df['property_form_numeric'].between(1, 11)
        invalid_count = invalid_mask.sum()
        if invalid_count > 0:
            rows_to_drop.update(df[invalid_mask].index.tolist())
            print(f"Удалено {invalid_count} строк с недопустимыми значениями в 'property_form_numeric' (должно быть от 1 до 11)")
    
    # Удаляем найденные строки
    if rows_to_drop:
        df_clean = df.drop(index=list(rows_to_drop))
        print(f"Всего удалено {len(rows_to_drop)} строк")
        print(f"Осталось {len(df_clean)} строк из исходных {initial_count}")
        return df_clean
    else:
        print("Недопустимых значений не найдено. Все строки остались без изменений.")
        return df
    
df = Del(df)

def calculate_descriptive_stats(df, numeric_columns):
    stats_df = pd.DataFrame(index=numeric_columns)
    
    for col in numeric_columns:
        if df[col].notna().sum() > 0:
            data = df[col].dropna()
            stats_df.loc[col, 'Количество'] = len(data)
            stats_df.loc[col, 'Среднее'] = data.mean()
            stats_df.loc[col, 'Дисперсия'] = data.var()
            stats_df.loc[col, 'Станд. отклонение'] = data.std()
            stats_df.loc[col, 'Мода'] = data.mode().iloc[0] if not data.mode().empty else np.nan
            stats_df.loc[col, 'Медиана'] = data.median()
            stats_df.loc[col, '1-й квартиль (Q1)'] = data.quantile(0.25)
            stats_df.loc[col, '3-й квартиль (Q3)'] = data.quantile(0.75)
            stats_df.loc[col, 'Коэф. вариации'] = (data.std() / data.mean()) * 100 if data.mean() != 0 else np.inf
            stats_df.loc[col, 'Асимметрия'] = stats.skew(data)
            stats_df.loc[col, 'Эксцесс'] = stats.kurtosis(data)
            stats_df.loc[col, 'Минимум'] = data.min()
            stats_df.loc[col, 'Максимум'] = data.max()
    
    return stats_df

# Статистики для числовых переменных
numeric_stats = calculate_descriptive_stats(df, numeric_columns)
print(numeric_stats.round(2))

# Статистики для категориальных переменных
categorical_columns = ['property_form_numeric', 'company_type_numeric']

for col in categorical_columns:
    if col in df.columns:
        print(f"\n{col}:")
        value_counts = df[col].value_counts()
        print("Распределение значений:")
        for value, count in value_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {value}: {count} ({percentage:.1f}%)")



def check_normality(df, numeric_columns):
    normality_results = []
    
    for col in numeric_columns:
        data = df[col].dropna()
            
        # a. Коэффициенты асимметрии и эксцесса
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        
        # b. Статистические тесты нормальности
        # Тест Шапиро-Уилка (для n < 5000)
        if len(data) <= 5000:
            shapiro_stat, shapiro_p = shapiro(data)
        else:
            shapiro_stat, shapiro_p = np.nan, np.nan
            
        # Тест Лиллиефорса
        lillie_stat, lillie_p = lilliefors(data, dist='norm')
            
        # Тест Харке-Бера (Jarque-Bera)
        jb_stat, jb_p = stats.jarque_bera(data)
            
        # Интерпретация результатов тестов
        alpha = 0.05
        tests_passed = 0
        total_tests = 3
            
        if not np.isnan(shapiro_p) and shapiro_p > alpha:
            tests_passed += 1
        if lillie_p > alpha:
            tests_passed += 1
        if jb_p > alpha:
            tests_passed += 1
            
        if tests_passed >= 2:
            normality_conclusion = "Нормальное"
        else:
            normality_conclusion = "Не нормальное"
            
        normality_results.append({
                'Переменная': col,
                'Асимметрия': round(skewness, 3),
                'Эксцесс': round(kurtosis, 3),
                'Shapiro-Wilk p-value': round(shapiro_p, 4) if not np.isnan(shapiro_p) else 'N/A',
                'Lilliefors p-value': round(lillie_p, 4),
                'Jarque-Bera p-value': round(jb_p, 4),
                'Вывод': normality_conclusion
            })
    
    return pd.DataFrame(normality_results)

# Проверка нормальности для основных переменных
normality_df = check_normality(df, numeric_columns)
print(normality_df.to_string(index=False))

# a. Гистограммы с наложением плотности нормального распределения
fig, axes = plt.subplots(4, 4, figsize=(15, 12))
axes = axes.ravel()

for i, col in enumerate(numeric_columns):
    if col in df.columns and df[col].notna().sum() > 0:
        data = df[col].dropna()
        
        # Гистограмма
        axes[i].hist(data, bins=15, density=True, alpha=0.6, color='skyblue', edgecolor='black')
        
        # Наложение нормальной кривой
        xmin, xmax = axes[i].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        if data.std() > 0:  # Избегаем деления на ноль
            p = norm.pdf(x, data.mean(), data.std())
            axes[i].plot(x, p, 'k', linewidth=2, label='Нормальное распределение')
        
        axes[i].set_title(f'{col}\nАсимметрия: {stats.skew(data):.2f}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Плотность')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

# Убираем лишние subplots
for i in range(len(numeric_columns), 16):
    axes[i].set_visible(False)

plt.tight_layout()
plt.show()

# b. Графики Квантиль-Квантиль (QQ-plot)

fig, axes = plt.subplots(4, 4, figsize=(15, 12))
axes = axes.ravel()

for i, col in enumerate(numeric_columns):
    if col in df.columns and df[col].notna().sum() > 0:
        data = df[col].dropna()
        
        # QQ-plot
        probplot(data, dist="norm", plot=axes[i])
        axes[i].set_title(f'QQ-plot для {col}')
        axes[i].grid(True, alpha=0.3)

# Убираем лишние subplots
for i in range(len(numeric_columns), 16):
    axes[i].set_visible(False)

plt.tight_layout()
plt.show()

##Классификация

numeric_features = [
    'revenue', 'net_profit', 'authorized_capital', 
    'contracts_count', 'customer_sum', 'supplier_sum',
    'inspections_count', 'inspections_violations', 'inspections_knm',
    'plaintiff_cases', 'defendant_cases', 'licenses_count', 
    'trademarks_count'
]

# Оставляем только строки без пропусков в выбранных признаках
df_cluster = df[numeric_features].dropna()
print(f"Количество записей для кластеризации: {len(df_cluster)}")

# Стандартизация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Метод локтя
inertia = []
k_range = range(2, 10)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Построение графика метода локтя
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, 'bo-')
plt.xlabel('Количество кластеров')
plt.ylabel('Inertia')
plt.title('Метод локтя для определения оптимального числа кластеров')
plt.grid(True)
plt.savefig('elbow_method.png', dpi=300, bbox_inches='tight')

# Метод силуэта
silhouette_scores = []
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"Количество кластеров: {k}, Silhouette Score: {score:.4f}")

# Построение графика силуэта
plt.figure(figsize=(10, 6))
plt.plot(k_range, silhouette_scores, 'go-')
plt.xlabel('Количество кластеров')
plt.ylabel('Silhouette Score')
plt.title('Метод силуэта для определения оптимального числа кластеров')
plt.grid(True)
plt.savefig('silhouette_method.png', dpi=300, bbox_inches='tight')

optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df_cluster['cluster'] = clusters

# Анализ размера кластеров
cluster_sizes = df_cluster['cluster'].value_counts().sort_index()
print("\nРазмеры кластеров:")
for cluster, size in cluster_sizes.items():
    print(f"Кластер {cluster}: {size} компаний ({size/len(df_cluster)*100:.2f}%)")

# Визуализация кластеров с помощью PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', 
                     s=50, alpha=0.6, edgecolor='w')
plt.xlabel('Первая главная компонента')
plt.ylabel('Вторая главная компонента')
plt.title('Визуализация кластеров с помощью PCA')
plt.colorbar(scatter, label='Кластер')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('clusters_pca.png', dpi=300, bbox_inches='tight')

# Тесты на нормальность для каждого кластера и каждой переменной
normality_results = []
alpha = 0.05

for cluster in range(optimal_k):
    cluster_data = df_cluster[df_cluster['cluster'] == cluster]
    
    print(f"\nКластер {cluster}:")
    for col in numeric_features:
        # Shapiro-Wilk test
        stat, p_value = stats.shapiro(cluster_data[col])
        is_normal = p_value > alpha
        
        normality_results.append({
            'cluster': cluster,
            'variable': col,
            'shapiro_p': p_value,
            'normal': is_normal
        })
        
        status = "нормальное" if is_normal else "не нормальное"
        print(f"  - {col}: p-value = {p_value:.4f} - {status}")
    
    # Процент нормально распределенных переменных в кластере
    normal_count = sum(1 for res in normality_results if res['cluster'] == cluster and res['normal'])
    print(f"  - Нормально распределенных переменных: {normal_count}/{len(numeric_features)}")


##Корреляционный анализ
def plot_correlation_fields(df, cluster_num, variables=None):
    if variables is None:
        variables = ['revenue', 'net_profit', 'authorized_capital', 'customer_sum', 'supplier_sum']
    
    cluster_data = df[df['cluster'] == cluster_num]
    
    # Выбираем только непрерывные переменные
    continuous_vars = []
    for var in variables:
        if var in cluster_data.columns and cluster_data[var].nunique() > 6:
            continuous_vars.append(var)
    
    # Матрица корреляций
    corr_matrix = cluster_data[continuous_vars].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title(f'Матрица корреляций: Кластер {cluster_num}')
    plt.tight_layout()
    plt.show()

    # Анализ гетероскедастичности
    print(f"\nАнализ гетероскедастичности для кластера {cluster_num }:")
    if 'revenue' in continuous_vars and 'net_profit' in continuous_vars:
        # Простой тест на гетероскедастичность: сравниваем дисперсии в разных диапазонах
        x = cluster_data['revenue']
        y = cluster_data['net_profit']
        
        # Делим данные на три группы по revenue
        low_mask = x <= x.quantile(0.33)
        mid_mask = (x > x.quantile(0.33)) & (x <= x.quantile(0.66))
        high_mask = x > x.quantile(0.66)
        
        var_low = y[low_mask].var()
        var_mid = y[mid_mask].var()
        var_high = y[high_mask].var()
        
        max_var = max(var_low, var_mid, var_high)
        min_var = min(var_low, var_mid, var_high)
        
        heteroscedasticity_ratio = max_var / min_var if min_var > 0 else np.inf
        
        if heteroscedasticity_ratio > 4:  # эвристический порог
            print("Возможна гетероскедастичность")
        else:
            print("Гомоскедастичность вероятна")

# Строим поля корреляции для каждого кластера
for cluster in sorted(df_cluster['cluster'].unique()):
    plot_correlation_fields(df_cluster, cluster)

# Создаем словарь для хранения результатов
quantitative_columns = ['revenue', 'net_profit', 'authorized_capital', 'customer_sum', 'supplier_sum'] 
qualitative_columns = [
    'contracts_count',
    'inspections_count', 'inspections_violations', 'inspections_knm',
    'plaintiff_cases', 'defendant_cases', 'licenses_count', 
    'trademarks_count'
]

# Анализ по кластерам
for cluster_num in range(3):

    # Данки текущего кластера
    cluster_data = df_cluster[df_cluster['cluster'] == cluster_num]
    
    # 1. КОРРЕЛЯЦИЯ ПИРСОНА ДЛЯ КОЛИЧЕСТВЕННЫХ ПЕРЕМЕННЫХ    
    # Создаем матрицы для корреляций и p-значений
    corr_matrix = cluster_data[quantitative_columns].corr(method='pearson')
    p_value_matrix = np.zeros((len(quantitative_columns), len(quantitative_columns)))
    
    # Заполняем матрицу p-значений
    for i, col1 in enumerate(quantitative_columns):
        for j, col2 in enumerate(quantitative_columns):
            if i != j:
                corr, p_value = pearsonr(cluster_data[col1].dropna(), cluster_data[col2].dropna())
                p_value_matrix[i, j] = p_value
    
    # Выводим результаты корреляций
    print("\nКоэффициенты корреляции Пирсона:")
    for i, col1 in enumerate(quantitative_columns):
        for j, col2 in enumerate(quantitative_columns):
            if i < j:  # Выводим только уникальные пары
                corr_value = corr_matrix.iloc[i, j]
                p_value = p_value_matrix[i, j]
                print(f"{col1} - {col2}: {corr_value:.3f} (p-value: {p_value:.4f})")
    
    # 2. ТЕПЛОВАЯ КАРТА КОРРЕЛЯЦИЙ ПИРСОНА
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Маска для верхнего треугольника
    
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True, 
                fmt='.3f',
                cbar_kws={'shrink': 0.8})
    
    plt.title(f'Тепловая карта корреляций Пирсона\nКластер {cluster_num} (n={len(cluster_data)})')
    plt.tight_layout()
    plt.show()
    
    # 3. АНАЛИЗ МУЛЬТИКОЛЛИНЕАРНОСТИ
    print(f"\nАнализ мультиколлинеарности:")
    high_corr_pairs = []
    for i, col1 in enumerate(quantitative_columns):
        for j, col2 in enumerate(quantitative_columns):
            if i < j and abs(corr_matrix.iloc[i, j]) > 0.7 and p_value_matrix[i, j] < 0.05:
                high_corr_pairs.append((col1, col2, corr_matrix.iloc[i, j]))
    
    if high_corr_pairs:
        print("Обнаружены высокие значимые корреляции (|r| > 0.7):")
        for pair in high_corr_pairs:
            print(f"  {pair[0]} - {pair[1]}: r = {pair[2]:.3f}")
        print("ВНИМАНИЕ: Возможна мультиколлинеарность!")
    else:
        print("Высоких корреляций не обнаружено")
    
    # 4. КОРРЕЛЯЦИЯ СПИРМЕНА ДЛЯ КАЧЕСТВЕННЫХ ПЕРЕМЕННЫХ
    if qualitative_columns:
        print(f"\nРанговые корреляции Спирмена (качественные vs количественные):")
        
        for qual_col in qualitative_columns:
            if qual_col in cluster_data.columns:
                print(f"\nКачественная переменная: {qual_col}")
                
                for quant_col in quantitative_columns:
                    # Удаляем пропущенные значения для пары переменных
                    valid_data = cluster_data[[qual_col, quant_col]].dropna()
                    
                    if len(valid_data) > 1:
                        corr_spearman, p_value_spearman = spearmanr(valid_data[qual_col], 
                                                                   valid_data[quant_col])                        
                        print(f"  {qual_col} - {quant_col}: ρ = {corr_spearman:.3f} (p-value: {p_value_spearman:.4f})")
                    else:
                        print(f"  {qual_col} - {quant_col}: недостаточно данных для анализа")


dependent_var = 'revenue'
independent_vars  = [
    'net_profit', 'authorized_capital', 'contracts_count', 
    'supplier_sum', 'inspections_count', 'plaintiff_cases',
    'defendant_cases', 'licenses_count', 'trademarks_count'
]

# Функция для расчета AIC и BIC
def calculate_aic_bic(y_true, y_pred, n_features):
    n = len(y_true)
    residuals = y_true - y_pred
    sse = np.sum(residuals**2)
    # AIC
    aic = n * np.log(sse / n) + 2 * n_features
    # BIC
    bic = n * np.log(sse / n) + n_features * np.log(n)
    return aic, bic

# Функция для расчета adjusted R²
def calculate_adj_r2(r2, n, k):
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

# Функция для F-теста
def f_test(y_true, y_pred, n_features):
    n = len(y_true)
    y_mean = np.mean(y_true)
    
    # Sum of squares
    sst = np.sum((y_true - y_mean)**2)  # Total sum of squares
    ssr = np.sum((y_pred - y_mean)**2)  # Regression sum of squares
    sse = np.sum((y_true - y_pred)**2)  # Error sum of squares
       
    # Degrees of freedom
    df_reg = n_features
    df_resid = n - n_features - 1
    
    # Mean squares
    msr = ssr / df_reg
    mse = sse / df_resid
    
    # F-statistic
    f_stat = msr / mse
    
    # p-value
    p_value = 1 - stats.f.cdf(f_stat, df_reg, df_resid)
    
    return f_stat, p_value

# Функция для теста Рамсея (RESET test)
def ramsey_reset_test(y_true, y_pred, X, alpha=0.05):
    try:
        n = len(y_true)
        # Добавляем квадраты предсказаний
        y_pred_sq = y_pred**2
        y_pred_cb = y_pred**3
        
        # Создаем расширенную матрицу признаков
        X_extended = np.column_stack([X, y_pred_sq, y_pred_cb])
        
        # Проверяем значимость добавленных членов
        model_extended = LinearRegression()
        model_extended.fit(X_extended, y_true)
        y_pred_extended = model_extended.predict(X_extended)
        
        # F-тест для сравнения моделей
        sse_reduced = np.sum((y_true - y_pred)**2)
        sse_full = np.sum((y_true - y_pred_extended)**2)
        
        df_reduced = X.shape[1]
        df_full = X_extended.shape[1]
        
        f_stat = ((sse_reduced - sse_full) / (df_full - df_reduced)) / (sse_full / (n - df_full - 1))
        p_value = 1 - stats.f.cdf(f_stat, df_full - df_reduced, n - df_full - 1)
        
        return f_stat, p_value
    except:
        return np.nan, np.nan
    
# Функция для критерия Зарембки (сравнение линейной и логарифмической моделей)
def zarembka_test(y_true, y_pred_linear, y_pred_log, alpha=0.05):
    try:
        n = len(y_true)
        
        # Остатки моделей
        residuals_linear = y_true - y_pred_linear
        residuals_log = y_true - y_pred_log
        
        # Суммы квадратов остатков
        rss_linear = np.sum(residuals_linear**2)
        rss_log = np.sum(residuals_log**2)
        
        # Статистика критерия Зарембки
        z_stat = (rss_log - rss_linear) / (rss_linear / n)
        
        # p-value (аппроксимация хи-квадрат распределением)
        p_value = 1 - stats.chi2.cdf(z_stat, 1)
        
        # Результат теста
        if p_value < alpha:
            result = "Линейная модель лучше"
        else:
            result = "Логарифмическая модель лучше"
            
        return z_stat, p_value, result, rss_linear, rss_log
        
    except:
        return np.nan, np.nan, "Ошибка расчета", np.nan, np.nan


results_summary = []
coefficients_summary = []

# Анализ по кластерам
for cluster_num in range(optimal_k):
    
    # Данные текущего кластера
    cluster_data = df_cluster[df_cluster['cluster'] == cluster_num].copy()
    
    # Разделяем на обучающую и тестовую выборки
    X = cluster_data[independent_vars].values
    y = cluster_data[dependent_var].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"Размер обучающей выборки: {len(X_train)}")
    print(f"Размер тестовой выборки: {len(X_test)}")
    
    # МОДЕЛЬ 1: Линейная регрессия
    try:
        model_linear = LinearRegression()
        model_linear.fit(X_train, y_train)
        y_pred_linear = model_linear.predict(X_test)

        # Создаем DataFrame с коэффициентами
        coef_df_linear = pd.DataFrame({
            'Переменная': ['Константа'] + independent_vars,
            'Коэффициент': [model_linear.intercept_] + list(model_linear.coef_)
        })

        # Добавляем статистическую значимость (t-статистики)
        # Для этого нам нужно вычислить стандартные ошибки вручную
        y_pred_linear = model_linear.predict(X_train)
        residuals = y_train - y_pred_linear
        mse = np.sum(residuals**2) / (len(y_train) - len(independent_vars) - 1)
        
        # Вычисляем стандартные ошибки коэффициентов
        X_train_const = np.column_stack([np.ones(len(X_train)), X_train])
        cov_matrix = mse * np.linalg.inv(X_train_const.T @ X_train_const)
        std_errors = np.sqrt(np.diag(cov_matrix))
        
        # t-статистики и p-values
        t_stats = coef_df_linear['Коэффициент'] / std_errors
        p_values = [2 * (1 - stats.t.cdf(np.abs(t), len(y_train) - len(independent_vars) - 1)) for t in t_stats]
        
        coef_df_linear['Стд. Ошибка'] = std_errors
        coef_df_linear['t-статистика'] = t_stats
        coef_df_linear['p-value'] = p_values
        
        print(coef_df_linear.round(4))
        
        # Метрики качества
        r2_linear = r2_score(y_test, y_pred_linear)
        mse_linear = mean_squared_error(y_test, y_pred_linear)
        mae_linear = mean_absolute_error(y_test, y_pred_linear)
        adj_r2_linear = calculate_adj_r2(r2_linear, len(y_test), X_test.shape[1])
        aic_linear, bic_linear = calculate_aic_bic(y_test, y_pred_linear, X_test.shape[1])
        f_stat_linear, f_pvalue_linear = f_test(y_test, y_pred_linear, X_test.shape[1])
        
        # Тест Рамсея
        reset_f_linear, reset_p_linear = ramsey_reset_test(y_test, y_pred_linear, X_test)
        
        print(f"R²: {r2_linear:.4f}")
        print(f"Adj R²: {adj_r2_linear:.4f}")
        print(f"MSE: {mse_linear:.4f}")
        print(f"MAE: {mae_linear:.4f}")
        print(f"AIC: {aic_linear:.4f}")
        print(f"BIC: {bic_linear:.4f}")
        print(f"F-статистика: {f_stat_linear:.4f}")
        print(f"p-value F-теста: {f_pvalue_linear:.4f}")
        print(f"Тест Рамсея p-value: {reset_p_linear:.4f}")
        
    except Exception as e:
        print(f"Ошибка в линейной модели: {e}")
        model_linear = None
        y_pred_linear = None
    
    # МОДЕЛЬ 2: Логарифмическая модель 
    try:
        # Применяем логарифм к целевой переменной
        y_train_log = np.log(y_train + 1e-6)
        
        model_log = LinearRegression()
        model_log.fit(X_train, y_train_log)
        y_pred_log = model_log.predict(X_test)

        # Коэффициенты для логарифмической модели
        coef_df_log = pd.DataFrame({
            'Переменная': ['Константа'] + independent_vars,
            'Коэффициент': [model_log.intercept_] + list(model_log.coef_)
        })
        
        # Вычисляем статистическую значимость
        y_pred_log = model_log.predict(X_train)
        residuals_log = y_train_log - y_pred_log
        mse_log = np.sum(residuals_log**2) / (len(y_train_log) - len(independent_vars) - 1)
        
        X_train_const = np.column_stack([np.ones(len(X_train)), X_train])
        cov_matrix_log = mse_log * np.linalg.inv(X_train_const.T @ X_train_const)
        std_errors_log = np.sqrt(np.diag(cov_matrix_log))
        
        t_stats_log = coef_df_log['Коэффициент'] / std_errors_log
        p_values_log = [2 * (1 - stats.t.cdf(np.abs(t), len(y_train_log) - len(independent_vars) - 1)) for t in t_stats_log]
        
        coef_df_log['Стд. Ошибка'] = std_errors_log
        coef_df_log['t-статистика'] = t_stats_log
        coef_df_log['p-value'] = p_values_log
        
        # Преобразуем обратно
        y_pred_log_exp = np.exp(y_pred_log) - 1e-6
        
        # Метрики качества
        r2_log = r2_score(y_test, y_pred_log_exp)
        mse_log = mean_squared_error(y_test, y_pred_log_exp)
        mae_log = mean_absolute_error(y_test, y_pred_log_exp)
        adj_r2_log = calculate_adj_r2(r2_log, len(y_test), X_test.shape[1])
        aic_log, bic_log = calculate_aic_bic(y_test, y_pred_log_exp, X_test.shape[1])
        f_stat_log, f_pvalue_log = f_test(y_test, y_pred_log_exp, X_test.shape[1])
        
        # Тест Рамсея
        reset_f_log, reset_p_log = ramsey_reset_test(y_test, y_pred_log_exp, X_test)
        
        print(f"R²: {r2_log:.4f}")
        print(f"Adj R²: {adj_r2_log:.4f}")
        print(f"MSE: {mse_log:.4f}")
        print(f"MAE: {mae_log:.4f}")
        print(f"AIC: {aic_log:.4f}")
        print(f"BIC: {bic_log:.4f}")
        print(f"F-статистика: {f_stat_log:.4f}")
        print(f"p-value F-теста: {f_pvalue_log:.4f}")
        print(f"Тест Рамсея p-value: {reset_p_log:.4f}")
        
    except Exception as e:
        print(f"Ошибка в логарифмической модели: {e}")
        model_log = None
        y_pred_log_exp = None
    
    # МОДЕЛЬ 3: Полиномиальная регрессия (степень 2)    
    try:
        # Создаем полиномиальные признаки
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        
        model_poly = LinearRegression()
        model_poly.fit(X_train_poly, y_train)
        y_pred_poly = model_poly.predict(X_test_poly)

        # Получаем имена полиномиальных признаков
        feature_names = poly.get_feature_names_out(independent_vars)
        
        # Коэффициенты для полиномиальной модели
        coef_df_poly = pd.DataFrame({
            'Переменная': ['Константа'] + list(feature_names),
            'Коэффициент': [model_poly.intercept_] + list(model_poly.coef_)
        })
        
        # Вычисляем статистическую значимость
        y_pred_poly = model_poly.predict(X_train_poly)
        residuals_poly = y_train - y_pred_poly
        mse_poly = np.sum(residuals_poly**2) / (len(y_train) - X_train_poly.shape[1] - 1)
        
        X_train_poly_const = np.column_stack([np.ones(len(X_train_poly)), X_train_poly])
        cov_matrix_poly = mse_poly * np.linalg.inv(X_train_poly_const.T @ X_train_poly_const)
        std_errors_poly = np.sqrt(np.diag(cov_matrix_poly))
        
        t_stats_poly = coef_df_poly['Коэффициент'] / std_errors_poly
        p_values_poly = [2 * (1 - stats.t.cdf(np.abs(t), len(y_train) - X_train_poly.shape[1] - 1)) for t in t_stats_poly]
        
        coef_df_poly['Стд. Ошибка'] = std_errors_poly
        coef_df_poly['t-статистика'] = t_stats_poly
        coef_df_poly['p-value'] = p_values_poly
        
        print(coef_df_poly.round(4))
        
        # Метрики качества
        r2_poly = r2_score(y_test, y_pred_poly)
        mse_poly = mean_squared_error(y_test, y_pred_poly)
        mae_poly = mean_absolute_error(y_test, y_pred_poly)
        adj_r2_poly = calculate_adj_r2(r2_poly, len(y_test), X_test_poly.shape[1])
        aic_poly, bic_poly = calculate_aic_bic(y_test, y_pred_poly, X_test_poly.shape[1])
        f_stat_poly, f_pvalue_poly = f_test(y_test, y_pred_poly, X_test_poly.shape[1])
        
        # Тест Рамсея
        reset_f_poly, reset_p_poly = ramsey_reset_test(y_test, y_pred_poly, X_test_poly)
        
        print(f"R²: {r2_poly:.4f}")
        print(f"Adj R²: {adj_r2_poly:.4f}")
        print(f"MSE: {mse_poly:.4f}")
        print(f"MAE: {mae_poly:.4f}")
        print(f"AIC: {aic_poly:.4f}")
        print(f"BIC: {bic_poly:.4f}")
        print(f"F-статистика: {f_stat_poly:.4f}")
        print(f"p-value F-теста: {f_pvalue_poly:.4f}")
        print(f"Тест Рамсея p-value: {reset_p_poly:.4f}")
        
    except Exception as e:
        print(f"Ошибка в полиномиальной модели: {e}")
        model_poly = None
    
    # МОДЕЛЬ 4: Ridge регрессия (регуляризация L2)
    try:
        model_ridge = Ridge(alpha=1.0)
        model_ridge.fit(X_train, y_train)
        y_pred_ridge = model_ridge.predict(X_test)

        # Коэффициенты для Ridge модели
        coef_df_ridge = pd.DataFrame({
            'Переменная': ['Константа'] + independent_vars,
            'Коэффициент': [model_ridge.intercept_] + list(model_ridge.coef_)
        })
        
        # Метрики качества
        r2_ridge = r2_score(y_test, y_pred_ridge)
        mse_ridge = mean_squared_error(y_test, y_pred_ridge)
        mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
        adj_r2_ridge = calculate_adj_r2(r2_ridge, len(y_test), X_test.shape[1])
        aic_ridge, bic_ridge = calculate_aic_bic(y_test, y_pred_ridge, X_test.shape[1])
        f_stat_ridge, f_pvalue_ridge = f_test(y_test, y_pred_ridge, X_test.shape[1])
        
        # Тест Рамсея
        reset_f_ridge, reset_p_ridge = ramsey_reset_test(y_test, y_pred_ridge, X_test)
        
        print(f"R²: {r2_ridge:.4f}")
        print(f"Adj R²: {adj_r2_ridge:.4f}")
        print(f"MSE: {mse_ridge:.4f}")
        print(f"MAE: {mae_ridge:.4f}")
        print(f"AIC: {aic_ridge:.4f}")
        print(f"BIC: {bic_ridge:.4f}")
        print(f"F-статистика: {f_stat_ridge:.4f}")
        print(f"p-value F-теста: {f_pvalue_ridge:.4f}")
        print(f"Тест Рамсея p-value: {reset_p_ridge:.4f}")
        
    except Exception as e:
        print(f"Ошибка в Ridge модели: {e}")
        model_ridge = None
    
    # КРИТЕРИЙ ЗАРЕМБКИ: Сравнение линейной и логарифмической моделей    
    if model_linear is not None and model_log is not None:
        z_stat, z_pvalue, z_result, rss_linear, rss_log = zarembka_test(
            y_test, y_pred_linear, y_pred_log_exp
        )
        
        print(f"Статистика Зарембки: {z_stat:.4f}")
        print(f"p-value критерия Зарембки: {z_pvalue:.4f}")
        print(f"Результат: {z_result}")
        print(f"RSS линейной модели: {rss_linear:.4f}")
        print(f"RSS логарифмической модели: {rss_log:.4f}")
        
        # Определяем победителя по критерию Зарембки
        if "Линейная" in z_result:
            zarembka_winner = "Линейная"
        else:
            zarembka_winner = "Логарифмическая"
    else:
        z_stat, z_pvalue, z_result, rss_linear, rss_log = np.nan, np.nan, "Недостаточно данных", np.nan, np.nan
        zarembka_winner = "Не определено"
    

best_models = {
    0: 'Ridge',
    1: 'Линейная', 
    2: 'Линейная'
}

#Тест Бреуша-Пагана на гетероскедастичность
def breusch_pagan_test(residuals, X):
    try:
        # Квадраты остатков
        residuals_sq = residuals ** 2
        
        # Регрессия квадратов остатков на независимые переменные
        X_with_const = np.column_stack([np.ones(len(X)), X])
        bp_model = LinearRegression()
        bp_model.fit(X_with_const, residuals_sq)
        residuals_sq_pred = bp_model.predict(X_with_const)
        
        # Объясненная сумма квадратов
        ess = np.sum((residuals_sq_pred - np.mean(residuals_sq)) ** 2)
        
        # Тестовая статистика
        bp_statistic = ess / 2  # делим на 2 т.к. дисперсия остатков
        
        # p-value из распределения хи-квадрат
        p_value = 1 - stats.chi2.cdf(bp_statistic, X.shape[1])
        
        return bp_statistic, p_value
    except:
        return np.nan, np.nan
    
# Функция для теста Уайта на гетероскедастичность
def white_test(residuals, X):
    try:
        # Квадраты остатков
        residuals_sq = residuals ** 2
        
        # Создаем матрицу с исходными переменными, их квадратами и взаимодействиями
        X_white = X.copy()
        
        # Добавляем квадраты переменных
        for i in range(X.shape[1]):
            X_white = np.column_stack([X_white, X[:, i] ** 2])
        
        # Добавляем попарные взаимодействия
        for i in range(X.shape[1]):
            for j in range(i+1, X.shape[1]):
                X_white = np.column_stack([X_white, X[:, i] * X[:, j]])
        
        # Регрессия квадратов остатков на расширенную матрицу
        X_white_const = np.column_stack([np.ones(len(X_white)), X_white])
        white_model = LinearRegression()
        white_model.fit(X_white_const, residuals_sq)
        residuals_sq_pred = white_model.predict(X_white_const)
        
        # Коэффициент детерминации
        r_sq = r2_score(residuals_sq, residuals_sq_pred)
        
        # Тестовая статистика
        white_statistic = len(residuals) * r_sq
        
        # Степени свободы (количество регрессоров в вспомогательной регрессии)
        df = X_white.shape[1]
        
        # p-value из распределения хи-квадрат
        p_value = 1 - stats.chi2.cdf(white_statistic, df)
        
        return white_statistic, p_value, df
    except:
        return np.nan, np.nan, np.nan

# Функция для теста Дарбина-Ватсона на автокорреляцию
def durbin_watson_test(residuals):
    try:
        diff = np.diff(residuals)
        dw_statistic = np.sum(diff ** 2) / np.sum(residuals ** 2)
        return dw_statistic
    except:
        return np.nan

# Функция для теста Льюнга-Бокса на автокорреляцию
def ljung_box_test(residuals, lags=10):
    try:
        n = len(residuals)
        q_statistic = 0
        for lag in range(1, lags + 1):
            autocorr = np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1]
            q_statistic += (autocorr ** 2) / (n - lag)
        q_statistic *= n * (n + 2)
        
        # p-value из распределения хи-квадрат
        p_value = 1 - stats.chi2.cdf(q_statistic, lags)
        
        return q_statistic, p_value
    except:
        return np.nan, np.nan

# Функция для расчета VIF (фактора инфляции дисперсии)
def calculate_vif(X, feature_names):
    try:
        vif_data = []
        for i, col_name in enumerate(feature_names):
            # Зависимая переменная - текущий признак
            y_vif = X[:, i]
            # Независимые переменные - все остальные признаки
            X_vif = np.delete(X, i, axis=1)
            
            # Обучаем регрессию
            vif_model = LinearRegression()
            vif_model.fit(X_vif, y_vif)
            y_pred_vif = vif_model.predict(X_vif)
            
            # R² для этой регрессии
            r2_vif = r2_score(y_vif, y_pred_vif)
            
            # VIF
            vif = 1 / (1 - r2_vif) if r2_vif < 1 else float('inf')
            
            vif_data.append({
                'Переменная': col_name,
                'VIF': vif
            })
        
        return pd.DataFrame(vif_data)
    except:
        return pd.DataFrame()

# Функция для теста Харке-Бера на нормальность
def jarque_bera_test(residuals):
    try:
        jb_statistic, jb_pvalue = jarque_bera(residuals)
        return jb_statistic, jb_pvalue
    except:
        return np.nan, np.nan

# Проверка условий для каждого кластера
gauss_markov_results = []

for cluster_num in range(optimal_k):
    
    # Данные текущего кластера
    cluster_data = df_cluster[df_cluster['cluster'] == cluster_num].copy()
    
    # Подготовка данных
    X = cluster_data[independent_vars].values
    y = cluster_data[dependent_var].values
    
    # Обучаем наилучшую модель для кластера
    if best_models[cluster_num] == 'Ridge':
        model = Ridge(alpha=1.0)
    else:  # Линейная модель
        model = LinearRegression()
    
    model.fit(X, y)
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # 8.1. ТЕСТ РАМСЕЯ НА ФУНКЦИОНАЛЬНУЮ ФОРМУ    
    reset_stat, reset_pvalue = ramsey_reset_test(y, y_pred, X)
    
    print(f"Статистика теста Рамсея: {reset_stat:.4f}")
    print(f"p-value: {reset_pvalue:.4f}")
    
    # 8.2. ГРАФИЧЕСКИЙ АНАЛИЗ ОСТАТКОВ    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # График 1: Остатки vs Предсказанные значения
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Предсказанные значения')
    axes[0, 0].set_ylabel('Остатки')
    axes[0, 0].set_title(f'Остатки vs Предсказанные значения\nКластер {cluster_num}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # График 2: Q-Q plot для нормальности
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title(f'Q-Q plot остатков\nКластер {cluster_num}')
    
    # График 3: Гистограмма распределения остатков
    axes[1, 0].hist(residuals, bins=20, density=True, alpha=0.7, color='skyblue')
    
    # Нормальное распределение для сравнения
    x_norm = np.linspace(residuals.min(), residuals.max(), 100)
    y_norm = stats.norm.pdf(x_norm, np.mean(residuals), np.std(residuals))
    axes[1, 0].plot(x_norm, y_norm, 'r-', linewidth=2)
    axes[1, 0].set_xlabel('Остатки')
    axes[1, 0].set_ylabel('Плотность')
    axes[1, 0].set_title(f'Распределение остатков\nКластер {cluster_num}')
    axes[1, 0].grid(True, alpha=0.3)
    
    # График 4: Автокорреляция остатков
    lags = min(20, len(residuals) - 1)
    autocorrelation = [np.corrcoef(residuals[:-i], residuals[i:])[0, 1] 
                      if i < len(residuals) else 0 for i in range(1, lags + 1)]
    
    axes[1, 1].bar(range(1, lags + 1), autocorrelation, alpha=0.7)
    axes[1, 1].axhline(y=0, color='r', linestyle='-')
    axes[1, 1].axhline(y=0.1, color='orange', linestyle='--', alpha=0.7)
    axes[1, 1].axhline(y=-0.1, color='orange', linestyle='--', alpha=0.7)
    axes[1, 1].set_xlabel('Лаг')
    axes[1, 1].set_ylabel('Автокорреляция')
    axes[1, 1].set_title(f'Автокорреляция остатков\nКластер {cluster_num}')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 8.3. ПРОВЕРКА РАВЕНСТВА НУЛЮ МАТЕМАТИЧЕСКОГО ОЖИДАНИЯ ОШИБОК    
    t_stat_mean, p_value_mean = stats.ttest_1samp(residuals, 0)
    
    print(f"t-статистика: {t_stat_mean:.4f}")
    print(f"p-value: {p_value_mean:.4f}")
    
    # 8.4. ПРОВЕРКА ГОМОСКЕДАСТИЧНОСТИ (отсутствия гетероскедастичности)
    # Тест Бреуша-Пагана
    bp_stat, bp_pvalue = breusch_pagan_test(residuals, X)
    print(f"Тест Бреуша-Пагана:")
    print(f"  Статистика: {bp_stat:.4f}")
    print(f"  p-value: {bp_pvalue:.4f}")
    
    # Тест Уайта
    white_stat, white_pvalue, white_df = white_test(residuals, X)
    print(f"Тест Уайта:")
    print(f"  Статистика: {white_stat:.4f}")
    print(f"  p-value: {white_pvalue:.4f}")
    print(f"  Степени свободы: {white_df}")
    
    # 8.5. ПРОВЕРКА АВТОКОРРЕЛЯЦИИ ОСТАТКОВ    
    # Тест Дарбина-Ватсона
    dw_statistic = durbin_watson_test(residuals)
    print(f"Тест Дарбина-Ватсона:")
    print(f"  Статистика: {dw_statistic:.4f}")
    
    # Тест Льюнга-Бокса
    lb_statistic, lb_pvalue = ljung_box_test(residuals)
    print(f"Тест Льюнга-Бокса:")
    print(f"  Статистика: {lb_statistic:.4f}")
    print(f"  p-value: {lb_pvalue:.4f}")
    
    # 8.6. ПРОВЕРКА МУЛЬТИКОЛЛИНЕАРНОСТИ
    
    vif_df = calculate_vif(X, independent_vars)
    
    if not vif_df.empty:
        print("VIF (Фактор инфляции дисперсии):")
        print(vif_df.round(4))
        
        # Проверяем на мультиколлинеарность
        high_vif_count = len(vif_df[vif_df['VIF'] > 10])
        moderate_vif_count = len(vif_df[(vif_df['VIF'] > 5) & (vif_df['VIF'] <= 10)])
    else:
        print("Не удалось рассчитать VIF")
        multicollinearity_result = "Ошибка"
    
    # 8.7. ПРОВЕРКА НОРМАЛЬНОСТИ РАСПРЕДЕЛЕНИЯ ОСТАТКОВ    
    # Тест Харке-Бера
    jb_statistic, jb_pvalue = jarque_bera_test(residuals)
    print(f"Тест Харке-Бера:")
    print(f"  Статистика: {jb_statistic:.4f}")
    print(f"  p-value: {jb_pvalue:.4f}")
    
    # Тест Шапиро-Уилка (для небольших выборок)
    if len(residuals) < 5000:
        shapiro_stat, shapiro_pvalue = shapiro(residuals)
        print(f"Тест Шапиро-Уилка:")
        print(f"  Статистика: {shapiro_stat:.4f}")
        print(f"  p-value: {shapiro_pvalue:.4f}")
    else:
        shapiro_stat, shapiro_pvalue = np.nan, np.nan
        print("Тест Шапиро-Уилка: не применим для n ≥ 5000")    