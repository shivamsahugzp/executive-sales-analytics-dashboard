-- Executive Sales Analytics - Key Metrics Query
-- This query provides comprehensive executive-level metrics for the sales dashboard

WITH sales_metrics AS (
    -- Core sales metrics
    SELECT 
        DATE_TRUNC('month', order_date) as sales_month,
        region_id,
        product_id,
        sales_rep_id,
        customer_id,
        SUM(revenue) as total_revenue,
        SUM(units_sold) as total_units,
        SUM(margin) as total_margin,
        COUNT(*) as total_orders,
        AVG(order_value) as avg_order_value,
        COUNT(DISTINCT customer_id) as unique_customers
    FROM sales
    WHERE order_date >= DATEADD(month, -24, GETDATE())
    GROUP BY 
        DATE_TRUNC('month', order_date),
        region_id,
        product_id,
        sales_rep_id,
        customer_id
),

monthly_totals AS (
    -- Monthly aggregated metrics
    SELECT 
        sales_month,
        SUM(total_revenue) as monthly_revenue,
        SUM(total_units) as monthly_units,
        SUM(total_margin) as monthly_margin,
        SUM(total_orders) as monthly_orders,
        AVG(avg_order_value) as monthly_avg_order_value,
        COUNT(DISTINCT unique_customers) as monthly_unique_customers
    FROM sales_metrics
    GROUP BY sales_month
),

regional_performance AS (
    -- Regional performance analysis
    SELECT 
        r.region_name,
        r.region_manager,
        r.region_code,
        SUM(sm.total_revenue) as regional_revenue,
        SUM(sm.total_units) as regional_units,
        SUM(sm.total_margin) as regional_margin,
        SUM(sm.total_orders) as regional_orders,
        AVG(sm.avg_order_value) as regional_avg_order_value,
        COUNT(DISTINCT sm.unique_customers) as regional_customers,
        COUNT(DISTINCT sm.sales_rep_id) as sales_reps_count,
        
        -- Performance ratios
        SUM(sm.total_revenue) / COUNT(DISTINCT sm.sales_rep_id) as revenue_per_rep,
        SUM(sm.total_revenue) / COUNT(DISTINCT sm.unique_customers) as revenue_per_customer,
        SUM(sm.total_margin) / SUM(sm.total_revenue) as margin_percentage,
        
        -- Growth calculations
        LAG(SUM(sm.total_revenue), 1) OVER (PARTITION BY r.region_name ORDER BY MAX(sm.sales_month)) as prev_month_revenue,
        LAG(SUM(sm.total_revenue), 12) OVER (PARTITION BY r.region_name ORDER BY MAX(sm.sales_month)) as prev_year_revenue
        
    FROM sales_metrics sm
    JOIN regions r ON sm.region_id = r.region_id
    GROUP BY r.region_name, r.region_manager, r.region_code
),

product_performance AS (
    -- Product performance analysis
    SELECT 
        p.category,
        p.product_name,
        p.product_code,
        p.unit_price,
        SUM(sm.total_revenue) as product_revenue,
        SUM(sm.total_units) as product_units,
        SUM(sm.total_margin) as product_margin,
        SUM(sm.total_orders) as product_orders,
        AVG(sm.avg_order_value) as product_avg_order_value,
        COUNT(DISTINCT sm.unique_customers) as product_customers,
        
        -- Product metrics
        SUM(sm.total_revenue) / SUM(sm.total_units) as revenue_per_unit,
        SUM(sm.total_margin) / SUM(sm.total_revenue) as product_margin_percentage,
        SUM(sm.total_units) / COUNT(DISTINCT sm.unique_customers) as units_per_customer,
        
        -- Ranking within category
        RANK() OVER (PARTITION BY p.category ORDER BY SUM(sm.total_revenue) DESC) as category_revenue_rank,
        RANK() OVER (PARTITION BY p.category ORDER BY SUM(sm.total_units) DESC) as category_units_rank
        
    FROM sales_metrics sm
    JOIN products p ON sm.product_id = p.product_id
    GROUP BY p.category, p.product_name, p.product_code, p.unit_price
),

sales_rep_performance AS (
    -- Sales representative performance
    SELECT 
        sr.sales_rep_name,
        sr.sales_rep_id,
        r.region_name,
        sr.hire_date,
        SUM(sm.total_revenue) as rep_revenue,
        SUM(sm.total_units) as rep_units,
        SUM(sm.total_margin) as rep_margin,
        SUM(sm.total_orders) as rep_orders,
        AVG(sm.avg_order_value) as rep_avg_order_value,
        COUNT(DISTINCT sm.unique_customers) as rep_customers,
        
        -- Performance metrics
        SUM(sm.total_revenue) / COUNT(DISTINCT sm.unique_customers) as revenue_per_customer,
        SUM(sm.total_orders) / COUNT(DISTINCT sm.unique_customers) as orders_per_customer,
        SUM(sm.total_margin) / SUM(sm.total_revenue) as rep_margin_percentage,
        
        -- Experience calculation
        DATEDIFF(month, sr.hire_date, GETDATE()) as months_experience,
        SUM(sm.total_revenue) / NULLIF(DATEDIFF(month, sr.hire_date, GETDATE()), 0) as revenue_per_month
        
    FROM sales_metrics sm
    JOIN sales_reps sr ON sm.sales_rep_id = sr.sales_rep_id
    JOIN regions r ON sr.region_id = r.region_id
    GROUP BY sr.sales_rep_name, sr.sales_rep_id, r.region_name, sr.hire_date
),

executive_kpis AS (
    -- Executive-level KPIs
    SELECT 
        -- Revenue KPIs
        SUM(mt.monthly_revenue) as total_revenue,
        AVG(mt.monthly_revenue) as avg_monthly_revenue,
        MAX(mt.monthly_revenue) as peak_monthly_revenue,
        MIN(mt.monthly_revenue) as lowest_monthly_revenue,
        
        -- Growth KPIs
        (MAX(mt.monthly_revenue) - MIN(mt.monthly_revenue)) / MIN(mt.monthly_revenue) as revenue_growth_rate,
        
        -- Unit KPIs
        SUM(mt.monthly_units) as total_units,
        AVG(mt.monthly_units) as avg_monthly_units,
        
        -- Margin KPIs
        SUM(mt.monthly_margin) as total_margin,
        AVG(mt.monthly_margin) as avg_monthly_margin,
        SUM(mt.monthly_margin) / SUM(mt.monthly_revenue) as overall_margin_percentage,
        
        -- Customer KPIs
        AVG(mt.monthly_unique_customers) as avg_monthly_customers,
        MAX(mt.monthly_unique_customers) as peak_monthly_customers,
        
        -- Order KPIs
        SUM(mt.monthly_orders) as total_orders,
        AVG(mt.monthly_orders) as avg_monthly_orders,
        AVG(mt.monthly_avg_order_value) as overall_avg_order_value
        
    FROM monthly_totals mt
),

trend_analysis AS (
    -- Trend analysis for forecasting
    SELECT 
        sales_month,
        monthly_revenue,
        monthly_units,
        monthly_margin,
        monthly_orders,
        monthly_unique_customers,
        
        -- Moving averages
        AVG(monthly_revenue) OVER (
            ORDER BY sales_month 
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) as revenue_3month_avg,
        
        AVG(monthly_revenue) OVER (
            ORDER BY sales_month 
            ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
        ) as revenue_6month_avg,
        
        AVG(monthly_revenue) OVER (
            ORDER BY sales_month 
            ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
        ) as revenue_12month_avg,
        
        -- Growth rates
        LAG(monthly_revenue, 1) OVER (ORDER BY sales_month) as prev_month_revenue,
        LAG(monthly_revenue, 12) OVER (ORDER BY sales_month) as prev_year_revenue,
        
        -- Month-over-month growth
        CASE 
            WHEN LAG(monthly_revenue, 1) OVER (ORDER BY sales_month) > 0 
            THEN (monthly_revenue - LAG(monthly_revenue, 1) OVER (ORDER BY sales_month)) / LAG(monthly_revenue, 1) OVER (ORDER BY sales_month)
            ELSE NULL 
        END as mom_growth_rate,
        
        -- Year-over-year growth
        CASE 
            WHEN LAG(monthly_revenue, 12) OVER (ORDER BY sales_month) > 0 
            THEN (monthly_revenue - LAG(monthly_revenue, 12) OVER (ORDER BY sales_month)) / LAG(monthly_revenue, 12) OVER (ORDER BY sales_month)
            ELSE NULL 
        END as yoy_growth_rate
        
    FROM monthly_totals
)

-- Final executive metrics output
SELECT 
    'EXECUTIVE_KPIS' as metric_category,
    NULL as region_name,
    NULL as product_name,
    NULL as sales_rep_name,
    NULL as sales_month,
    kpis.total_revenue,
    kpis.avg_monthly_revenue,
    kpis.revenue_growth_rate,
    kpis.total_units,
    kpis.total_margin,
    kpis.overall_margin_percentage,
    kpis.total_orders,
    kpis.overall_avg_order_value,
    kpis.avg_monthly_customers,
    NULL as revenue_per_rep,
    NULL as revenue_per_customer,
    NULL as margin_percentage,
    NULL as mom_growth_rate,
    NULL as yoy_growth_rate,
    NULL as category_revenue_rank,
    NULL as months_experience,
    GETDATE() as analysis_timestamp

FROM executive_kpis kpis

UNION ALL

-- Regional performance metrics
SELECT 
    'REGIONAL_PERFORMANCE' as metric_category,
    rp.region_name,
    NULL as product_name,
    NULL as sales_rep_name,
    NULL as sales_month,
    rp.regional_revenue as total_revenue,
    NULL as avg_monthly_revenue,
    CASE 
        WHEN rp.prev_year_revenue > 0 
        THEN (rp.regional_revenue - rp.prev_year_revenue) / rp.prev_year_revenue
        ELSE NULL 
    END as revenue_growth_rate,
    rp.regional_units as total_units,
    rp.regional_margin as total_margin,
    rp.margin_percentage as overall_margin_percentage,
    rp.regional_orders as total_orders,
    rp.regional_avg_order_value as overall_avg_order_value,
    rp.regional_customers as avg_monthly_customers,
    rp.revenue_per_rep,
    rp.revenue_per_customer,
    rp.margin_percentage,
    NULL as mom_growth_rate,
    NULL as yoy_growth_rate,
    NULL as category_revenue_rank,
    NULL as months_experience,
    GETDATE() as analysis_timestamp

FROM regional_performance rp

UNION ALL

-- Product performance metrics
SELECT 
    'PRODUCT_PERFORMANCE' as metric_category,
    NULL as region_name,
    pp.product_name,
    NULL as sales_rep_name,
    NULL as sales_month,
    pp.product_revenue as total_revenue,
    NULL as avg_monthly_revenue,
    NULL as revenue_growth_rate,
    pp.product_units as total_units,
    pp.product_margin as total_margin,
    pp.product_margin_percentage as overall_margin_percentage,
    pp.product_orders as total_orders,
    pp.product_avg_order_value as overall_avg_order_value,
    pp.product_customers as avg_monthly_customers,
    NULL as revenue_per_rep,
    pp.revenue_per_unit as revenue_per_customer,
    pp.product_margin_percentage as margin_percentage,
    NULL as mom_growth_rate,
    NULL as yoy_growth_rate,
    pp.category_revenue_rank,
    NULL as months_experience,
    GETDATE() as analysis_timestamp

FROM product_performance pp

UNION ALL

-- Sales rep performance metrics
SELECT 
    'SALES_REP_PERFORMANCE' as metric_category,
    srp.region_name,
    NULL as product_name,
    srp.sales_rep_name,
    NULL as sales_month,
    srp.rep_revenue as total_revenue,
    NULL as avg_monthly_revenue,
    NULL as revenue_growth_rate,
    srp.rep_units as total_units,
    srp.rep_margin as total_margin,
    srp.rep_margin_percentage as overall_margin_percentage,
    srp.rep_orders as total_orders,
    srp.rep_avg_order_value as overall_avg_order_value,
    srp.rep_customers as avg_monthly_customers,
    srp.revenue_per_customer as revenue_per_rep,
    srp.revenue_per_customer,
    srp.rep_margin_percentage as margin_percentage,
    NULL as mom_growth_rate,
    NULL as yoy_growth_rate,
    NULL as category_revenue_rank,
    srp.months_experience,
    GETDATE() as analysis_timestamp

FROM sales_rep_performance srp

UNION ALL

-- Trend analysis metrics
SELECT 
    'TREND_ANALYSIS' as metric_category,
    NULL as region_name,
    NULL as product_name,
    NULL as sales_rep_name,
    ta.sales_month,
    ta.monthly_revenue as total_revenue,
    ta.revenue_3month_avg as avg_monthly_revenue,
    ta.mom_growth_rate as revenue_growth_rate,
    ta.monthly_units as total_units,
    ta.monthly_margin as total_margin,
    ta.monthly_margin / NULLIF(ta.monthly_revenue, 0) as overall_margin_percentage,
    ta.monthly_orders as total_orders,
    NULL as overall_avg_order_value,
    ta.monthly_unique_customers as avg_monthly_customers,
    NULL as revenue_per_rep,
    NULL as revenue_per_customer,
    NULL as margin_percentage,
    ta.mom_growth_rate,
    ta.yoy_growth_rate,
    NULL as category_revenue_rank,
    NULL as months_experience,
    GETDATE() as analysis_timestamp

FROM trend_analysis ta

ORDER BY metric_category, total_revenue DESC;
