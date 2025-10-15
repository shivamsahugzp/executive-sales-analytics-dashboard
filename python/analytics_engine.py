"""
Executive Sales Analytics Engine
Advanced analytics engine for executive sales dashboard
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class ExecutiveSalesAnalytics:
    """
    Advanced sales analytics engine for executive insights and decision making
    """
    
    def __init__(self, data_source=None, data=None):
        """
        Initialize the analytics engine
        
        Args:
            data_source (str): Path to data source
            data (pd.DataFrame): Pre-loaded data
        """
        if data is not None:
            self.data = data
        elif data_source:
            self.data = pd.read_csv(data_source)
        else:
            raise ValueError("Either data_source or data must be provided")
        
        self.models = {}
        self.insights = []
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare and clean data for analysis"""
        # Convert date columns
        if 'order_date' in self.data.columns:
            self.data['order_date'] = pd.to_datetime(self.data['order_date'])
            self.data['year'] = self.data['order_date'].dt.year
            self.data['month'] = self.data['order_date'].dt.month
            self.data['quarter'] = self.data['order_date'].dt.quarter
            self.data['day_of_week'] = self.data['order_date'].dt.dayofweek
        
        # Calculate derived metrics
        if 'revenue' in self.data.columns and 'units_sold' in self.data.columns:
            self.data['revenue_per_unit'] = self.data['revenue'] / self.data['units_sold']
        
        if 'revenue' in self.data.columns and 'cost' in self.data.columns:
            self.data['margin'] = self.data['revenue'] - self.data['cost']
            self.data['margin_percentage'] = (self.data['margin'] / self.data['revenue']) * 100
    
    def calculate_executive_kpis(self):
        """
        Calculate key performance indicators for executive dashboard
        
        Returns:
            dict: Dictionary of executive KPIs
        """
        kpis = {}
        
        # Revenue KPIs
        kpis['total_revenue'] = self.data['revenue'].sum()
        kpis['avg_monthly_revenue'] = self.data.groupby(['year', 'month'])['revenue'].sum().mean()
        kpis['revenue_growth_yoy'] = self._calculate_yoy_growth('revenue')
        kpis['revenue_growth_mom'] = self._calculate_mom_growth('revenue')
        
        # Unit KPIs
        kpis['total_units'] = self.data['units_sold'].sum()
        kpis['avg_order_value'] = self.data['revenue'].sum() / self.data['order_id'].nunique()
        kpis['units_per_order'] = self.data['units_sold'].sum() / self.data['order_id'].nunique()
        
        # Customer KPIs
        kpis['total_customers'] = self.data['customer_id'].nunique()
        kpis['avg_revenue_per_customer'] = kpis['total_revenue'] / kpis['total_customers']
        kpis['customer_retention_rate'] = self._calculate_retention_rate()
        
        # Margin KPIs
        if 'margin' in self.data.columns:
            kpis['total_margin'] = self.data['margin'].sum()
            kpis['avg_margin_percentage'] = self.data['margin_percentage'].mean()
            kpis['margin_growth_yoy'] = self._calculate_yoy_growth('margin')
        
        # Regional KPIs
        if 'region' in self.data.columns:
            regional_revenue = self.data.groupby('region')['revenue'].sum()
            kpis['top_region'] = regional_revenue.idxmax()
            kpis['top_region_revenue'] = regional_revenue.max()
            kpis['regional_concentration'] = regional_revenue.max() / regional_revenue.sum()
        
        # Product KPIs
        if 'product_category' in self.data.columns:
            category_revenue = self.data.groupby('product_category')['revenue'].sum()
            kpis['top_category'] = category_revenue.idxmax()
            kpis['top_category_revenue'] = category_revenue.max()
            kpis['category_diversity'] = len(category_revenue)
        
        return kpis
    
    def analyze_regional_performance(self):
        """
        Analyze regional sales performance
        
        Returns:
            pd.DataFrame: Regional performance analysis
        """
        if 'region' not in self.data.columns:
            return pd.DataFrame()
        
        regional_analysis = self.data.groupby('region').agg({
            'revenue': ['sum', 'mean', 'std'],
            'units_sold': 'sum',
            'customer_id': 'nunique',
            'order_id': 'nunique',
            'margin': 'sum' if 'margin' in self.data.columns else lambda x: 0
        }).round(2)
        
        # Flatten column names
        regional_analysis.columns = ['_'.join(col).strip() for col in regional_analysis.columns]
        
        # Calculate performance metrics
        regional_analysis['revenue_per_customer'] = (
            regional_analysis['revenue_sum'] / regional_analysis['customer_id_nunique']
        )
        regional_analysis['orders_per_customer'] = (
            regional_analysis['order_id_nunique'] / regional_analysis['customer_id_nunique']
        )
        regional_analysis['avg_order_value'] = (
            regional_analysis['revenue_sum'] / regional_analysis['order_id_nunique']
        )
        
        if 'margin_sum' in regional_analysis.columns:
            regional_analysis['margin_percentage'] = (
                regional_analysis['margin_sum'] / regional_analysis['revenue_sum'] * 100
            )
        
        # Calculate growth rates
        regional_analysis['revenue_growth'] = self._calculate_regional_growth('revenue')
        
        # Performance ranking
        regional_analysis['revenue_rank'] = regional_analysis['revenue_sum'].rank(ascending=False)
        regional_analysis['growth_rank'] = regional_analysis['revenue_growth'].rank(ascending=False)
        
        return regional_analysis.sort_values('revenue_sum', ascending=False)
    
    def analyze_product_performance(self):
        """
        Analyze product performance across categories
        
        Returns:
            pd.DataFrame: Product performance analysis
        """
        if 'product_category' not in self.data.columns:
            return pd.DataFrame()
        
        product_analysis = self.data.groupby(['product_category', 'product_name']).agg({
            'revenue': ['sum', 'mean'],
            'units_sold': 'sum',
            'customer_id': 'nunique',
            'order_id': 'nunique',
            'margin': 'sum' if 'margin' in self.data.columns else lambda x: 0
        }).round(2)
        
        # Flatten column names
        product_analysis.columns = ['_'.join(col).strip() for col in product_analysis.columns]
        
        # Calculate product metrics
        product_analysis['revenue_per_unit'] = (
            product_analysis['revenue_sum'] / product_analysis['units_sold_sum']
        )
        product_analysis['units_per_customer'] = (
            product_analysis['units_sold_sum'] / product_analysis['customer_id_nunique']
        )
        product_analysis['avg_order_value'] = (
            product_analysis['revenue_sum'] / product_analysis['order_id_nunique']
        )
        
        if 'margin_sum' in product_analysis.columns:
            product_analysis['margin_percentage'] = (
                product_analysis['margin_sum'] / product_analysis['revenue_sum'] * 100
            )
        
        # Category-level analysis
        category_analysis = self.data.groupby('product_category').agg({
            'revenue': 'sum',
            'units_sold': 'sum',
            'customer_id': 'nunique'
        })
        
        category_analysis['category_share'] = (
            category_analysis['revenue'] / category_analysis['revenue'].sum() * 100
        )
        
        # Add category rankings
        product_analysis['category_revenue_rank'] = product_analysis.groupby('product_category')['revenue_sum'].rank(ascending=False)
        
        return product_analysis.sort_values('revenue_sum', ascending=False)
    
    def forecast_revenue(self, months_ahead=6, method='linear'):
        """
        Forecast revenue for future months
        
        Args:
            months_ahead (int): Number of months to forecast
            method (str): Forecasting method ('linear', 'random_forest')
        
        Returns:
            dict: Forecast results
        """
        # Prepare time series data
        monthly_data = self.data.groupby(['year', 'month'])['revenue'].sum().reset_index()
        monthly_data['date'] = pd.to_datetime(monthly_data[['year', 'month']].assign(day=1))
        monthly_data = monthly_data.sort_values('date')
        
        # Create features
        monthly_data['month_num'] = range(len(monthly_data))
        monthly_data['trend'] = monthly_data['month_num']
        monthly_data['seasonal'] = monthly_data['month'].apply(lambda x: np.sin(2 * np.pi * x / 12))
        
        X = monthly_data[['trend', 'seasonal']].values
        y = monthly_data['revenue'].values
        
        # Train model
        if method == 'linear':
            model = LinearRegression()
        elif method == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError("Method must be 'linear' or 'random_forest'")
        
        model.fit(X, y)
        self.models['revenue_forecast'] = model
        
        # Generate forecast
        last_month = monthly_data['month_num'].max()
        future_months = np.arange(last_month + 1, last_month + months_ahead + 1)
        future_seasonal = np.sin(2 * np.pi * (future_months % 12) / 12)
        X_future = np.column_stack([future_months, future_seasonal])
        
        forecast = model.predict(X_future)
        
        # Calculate confidence intervals
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        confidence_interval = 1.96 * mae  # 95% confidence interval
        
        # Create forecast dates
        last_date = monthly_data['date'].max()
        forecast_dates = [last_date + timedelta(days=30*i) for i in range(1, months_ahead + 1)]
        
        return {
            'forecast_dates': forecast_dates,
            'forecast_values': forecast,
            'confidence_interval': confidence_interval,
            'model_accuracy': 1 - (mae / y.mean()),
            'method': method
        }
    
    def generate_executive_insights(self):
        """
        Generate key insights for executive decision making
        
        Returns:
            list: List of executive insights
        """
        insights = []
        
        # Calculate KPIs
        kpis = self.calculate_executive_kpis()
        
        # Revenue insights
        if kpis['revenue_growth_yoy'] > 0.15:
            insights.append(f"Strong revenue growth: {kpis['revenue_growth_yoy']:.1%} YoY - consider expansion opportunities")
        elif kpis['revenue_growth_yoy'] < -0.05:
            insights.append(f"Revenue declining: {kpis['revenue_growth_yoy']:.1%} YoY - immediate action required")
        else:
            insights.append(f"Stable revenue growth: {kpis['revenue_growth_yoy']:.1%} YoY")
        
        # Regional insights
        regional_perf = self.analyze_regional_performance()
        if not regional_perf.empty:
            top_region = regional_perf.index[0]
            bottom_region = regional_perf.index[-1]
            insights.append(f"Top performing region: {top_region} (${regional_perf.loc[top_region, 'revenue_sum']:,.0f})")
            insights.append(f"Region needing attention: {bottom_region} (${regional_perf.loc[bottom_region, 'revenue_sum']:,.0f})")
        
        # Product insights
        product_perf = self.analyze_product_performance()
        if not product_perf.empty:
            top_category = product_perf.index[0][0]  # First level of multi-index
            insights.append(f"Best performing category: {top_category}")
        
        # Customer insights
        if kpis['customer_retention_rate'] > 0.8:
            insights.append(f"Excellent customer retention: {kpis['customer_retention_rate']:.1%}")
        elif kpis['customer_retention_rate'] < 0.6:
            insights.append(f"Low customer retention: {kpis['customer_retention_rate']:.1%} - focus on retention strategies")
        
        # Margin insights
        if 'avg_margin_percentage' in kpis:
            if kpis['avg_margin_percentage'] > 25:
                insights.append(f"Strong margins: {kpis['avg_margin_percentage']:.1f}% - maintain pricing strategy")
            elif kpis['avg_margin_percentage'] < 15:
                insights.append(f"Low margins: {kpis['avg_margin_percentage']:.1f}% - review pricing and costs")
        
        self.insights = insights
        return insights
    
    def create_executive_dashboard_data(self):
        """
        Create data structure for executive dashboard
        
        Returns:
            dict: Dashboard data structure
        """
        kpis = self.calculate_executive_kpis()
        regional_perf = self.analyze_regional_performance()
        product_perf = self.analyze_product_performance()
        insights = self.generate_executive_insights()
        
        # Revenue forecast
        try:
            forecast = self.forecast_revenue(months_ahead=6)
        except:
            forecast = None
        
        return {
            'kpis': kpis,
            'regional_performance': regional_perf.to_dict('index') if not regional_perf.empty else {},
            'product_performance': product_perf.to_dict('index') if not product_perf.empty else {},
            'insights': insights,
            'forecast': forecast,
            'last_updated': datetime.now().isoformat()
        }
    
    def _calculate_yoy_growth(self, metric):
        """Calculate year-over-year growth rate"""
        if 'year' not in self.data.columns:
            return 0
        
        current_year = self.data['year'].max()
        previous_year = current_year - 1
        
        current_value = self.data[self.data['year'] == current_year][metric].sum()
        previous_value = self.data[self.data['year'] == previous_year][metric].sum()
        
        if previous_value == 0:
            return 0
        
        return (current_value - previous_value) / previous_value
    
    def _calculate_mom_growth(self, metric):
        """Calculate month-over-month growth rate"""
        if 'month' not in self.data.columns or 'year' not in self.data.columns:
            return 0
        
        monthly_data = self.data.groupby(['year', 'month'])[metric].sum()
        if len(monthly_data) < 2:
            return 0
        
        current_value = monthly_data.iloc[-1]
        previous_value = monthly_data.iloc[-2]
        
        if previous_value == 0:
            return 0
        
        return (current_value - previous_value) / previous_value
    
    def _calculate_retention_rate(self):
        """Calculate customer retention rate"""
        if 'customer_id' not in self.data.columns or 'order_date' not in self.data.columns:
            return 0
        
        # Get customers from first half of period
        mid_point = self.data['order_date'].median()
        first_half_customers = set(
            self.data[self.data['order_date'] <= mid_point]['customer_id'].unique()
        )
        
        # Get customers from second half of period
        second_half_customers = set(
            self.data[self.data['order_date'] > mid_point]['customer_id'].unique()
        )
        
        if len(first_half_customers) == 0:
            return 0
        
        retained_customers = first_half_customers.intersection(second_half_customers)
        return len(retained_customers) / len(first_half_customers)
    
    def _calculate_regional_growth(self, metric):
        """Calculate regional growth rates"""
        if 'region' not in self.data.columns or 'year' not in self.data.columns:
            return pd.Series()
        
        regional_growth = {}
        for region in self.data['region'].unique():
            region_data = self.data[self.data['region'] == region]
            growth = self._calculate_yoy_growth_for_region(region_data, metric)
            regional_growth[region] = growth
        
        return pd.Series(regional_growth)
    
    def _calculate_yoy_growth_for_region(self, region_data, metric):
        """Calculate YoY growth for a specific region"""
        if 'year' not in region_data.columns:
            return 0
        
        current_year = region_data['year'].max()
        previous_year = current_year - 1
        
        current_value = region_data[region_data['year'] == current_year][metric].sum()
        previous_value = region_data[region_data['year'] == previous_year][metric].sum()
        
        if previous_value == 0:
            return 0
        
        return (current_value - previous_value) / previous_value


def create_sample_data():
    """Create sample data for testing the analytics engine"""
    np.random.seed(42)
    
    # Generate sample data
    n_records = 10000
    regions = ['North', 'South', 'East', 'West', 'Central']
    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books']
    products = [f'Product_{i}' for i in range(1, 51)]
    
    data = {
        'order_id': [f'ORD_{i:06d}' for i in range(1, n_records + 1)],
        'customer_id': [f'CUST_{np.random.randint(1, 1001):04d}' for _ in range(n_records)],
        'product_id': [f'PROD_{np.random.randint(1, 51):03d}' for _ in range(n_records)],
        'product_name': np.random.choice(products, n_records),
        'product_category': np.random.choice(categories, n_records),
        'region': np.random.choice(regions, n_records),
        'sales_rep_id': [f'REP_{np.random.randint(1, 21):03d}' for _ in range(n_records)],
        'order_date': pd.date_range('2022-01-01', '2024-12-31', periods=n_records),
        'units_sold': np.random.randint(1, 10, n_records),
        'unit_price': np.random.uniform(10, 500, n_records),
        'cost': np.random.uniform(5, 250, n_records)
    }
    
    # Calculate revenue
    data['revenue'] = data['units_sold'] * data['unit_price']
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Create sample data and run analytics
    sample_data = create_sample_data()
    analytics = ExecutiveSalesAnalytics(data=sample_data)
    
    # Generate insights
    insights = analytics.generate_executive_insights()
    print("Executive Insights:")
    for insight in insights:
        print(f"- {insight}")
    
    # Create dashboard data
    dashboard_data = analytics.create_executive_dashboard_data()
    print(f"\nDashboard data created with {len(dashboard_data['kpis'])} KPIs")
