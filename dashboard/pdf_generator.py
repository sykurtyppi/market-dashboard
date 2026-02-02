"""
PDF Report Generator for Market Dashboard
Generates professional weekly market reports
"""

import os
import logging
from datetime import datetime
from pathlib import Path
import plotly.io as pio
import pandas as pd
from fpdf import FPDF

logger = logging.getLogger(__name__)


class MarketReportPDF(FPDF):
    """Custom PDF class with header/footer"""
    
    def __init__(self):
        super().__init__()
        self.report_date = datetime.now().strftime("%B %d, %Y")
    
    def header(self):
        """Add header to each page"""
        self.set_font('Arial', 'B', 16)
        self.set_text_color(31, 119, 180)  # Blue color
        self.cell(0, 10, 'Market Risk Dashboard', 0, 1, 'C')
        self.set_font('Arial', '', 10)
        self.set_text_color(128, 128, 128)  # Gray
        self.cell(0, 5, f'Weekly Report - {self.report_date}', 0, 1, 'C')
        self.ln(5)
    
    def footer(self):
        """Add footer to each page"""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()} | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 0, 'C')
    
    def chapter_title(self, title):
        """Add a chapter title"""
        self.set_font('Arial', 'B', 14)
        self.set_text_color(0, 0, 0)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2)
    
    def add_metric_card(self, title, value, status_color='neutral'):
        """Add a metric card"""
        colors = {
            'good': (76, 175, 80),
            'warning': (255, 152, 0),
            'bad': (244, 67, 54),
            'neutral': (158, 158, 158)
        }
        
        color = colors.get(status_color, colors['neutral'])
        
        # Draw colored box
        self.set_fill_color(*color)
        self.rect(self.get_x(), self.get_y(), 90, 25, 'F')
        
        # Add text
        self.set_text_color(255, 255, 255)
        self.set_font('Arial', 'B', 10)
        self.cell(90, 10, title, 0, 0, 'C')
        self.ln(10)
        
        self.set_font('Arial', 'B', 14)
        self.cell(90, 10, str(value), 0, 1, 'C')
        self.ln(5)


class PDFReportGenerator:
    """Generate PDF reports from dashboard data"""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir = Path("/tmp/market_dashboard_charts")
        self.temp_dir.mkdir(exist_ok=True)
    
    def save_plotly_chart(self, fig, name: str) -> str:
        """Save Plotly figure as PNG"""
        filepath = self.temp_dir / f"{name}.png"
        try:
            pio.write_image(fig, str(filepath), width=800, height=400, scale=2)
            logger.info(f"Saved chart: {name}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to save chart {name}: {e}")
            return None
    
    def generate_report(
        self,
        snapshot: dict,
        vrp_data: dict,
        charts: dict,
        filename: str = None
    ) -> str:
        """
        Generate PDF report
        
        Args:
            snapshot: Latest market snapshot from database
            vrp_data: VRP analysis data
            charts: Dictionary of Plotly figures {name: fig}
            filename: Custom filename (optional)
        
        Returns:
            Path to generated PDF file
        """
        if filename is None:
            filename = f"market_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        
        output_path = self.output_dir / filename
        
        # Initialize PDF
        pdf = MarketReportPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Page 1: Executive Summary
        pdf.add_page()
        pdf.chapter_title("EXECUTIVE SUMMARY")
        
        # Market Regime Summary
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 6, 
            "This report provides a comprehensive overview of current market conditions, "
            "including credit spreads, volatility analysis, liquidity metrics, and sentiment indicators."
        )
        pdf.ln(5)
        
        # Key Metrics Grid
        pdf.chapter_title("KEY INDICATORS")
        
        # Row 1: Credit & VIX
        x_start = pdf.get_x()
        y_start = pdf.get_y()
        
        if snapshot and snapshot.get('credit_spread_hy'):
            hy_spread = snapshot['credit_spread_hy'] * 100
            color = 'good' if hy_spread < 300 else 'warning' if hy_spread < 450 else 'bad'
            pdf.add_metric_card("HYG Spread", f"{hy_spread:.0f} bps", color)
        
        pdf.set_xy(x_start + 100, y_start)
        
        if snapshot and snapshot.get('vix_spot'):
            vix = snapshot['vix_spot']
            color = 'good' if vix < 20 else 'warning' if vix < 30 else 'bad'
            pdf.add_metric_card("VIX", f"{vix:.1f}", color)
        
        # Row 2: VRP & Fear/Greed
        y_start = pdf.get_y()
        pdf.set_xy(x_start, y_start)
        
        if vrp_data and vrp_data.get('vrp'):
            vrp = vrp_data['vrp']
            color = 'good' if vrp > 4 else 'warning' if vrp > 0 else 'bad'
            pdf.add_metric_card("VRP", f"{vrp:+.2f}", color)
        
        pdf.set_xy(x_start + 100, y_start)
        
        if snapshot and snapshot.get('fear_greed_score'):
            fg = snapshot['fear_greed_score']
            color = 'bad' if fg < 25 else 'warning' if fg < 45 else 'good' if fg < 75 else 'warning'
            pdf.add_metric_card("Fear & Greed", f"{fg:.0f}", color)
        
        # Page 2: Charts
        if charts:
            pdf.add_page()
            pdf.chapter_title("MARKET ANALYSIS")
            
            # Save and add charts
            chart_num = 0
            for chart_name, fig in charts.items():
                if chart_num > 0 and chart_num % 2 == 0:
                    pdf.add_page()
                
                img_path = self.save_plotly_chart(fig, chart_name)
                if img_path and os.path.exists(img_path):
                    try:
                        # Add chart title
                        pdf.set_font('Arial', 'B', 11)
                        pdf.cell(0, 8, chart_name.replace('_', ' ').title(), 0, 1)
                        
                        # Add chart image
                        pdf.image(img_path, x=10, w=190)
                        pdf.ln(5)
                        chart_num += 1
                    except Exception as e:
                        logger.error(f"Failed to add chart {chart_name}: {e}")
        
        # Page 3: Data Summary
        pdf.add_page()
        pdf.chapter_title("MARKET DATA SUMMARY")
        
        pdf.set_font('Arial', '', 9)
        
        # Create data table
        if snapshot:
            data_items = [
                ("Date", snapshot.get('date', 'N/A')),
                ("HYG Spread", f"{snapshot.get('credit_spread_hy', 0)*100:.0f} bps"),
                ("LQD Spread", f"{snapshot.get('credit_spread_ig', 0)*100:.0f} bps"),
                ("10Y Treasury", f"{snapshot.get('treasury_10y', 0):.2f}%"),
                ("VIX", f"{snapshot.get('vix_spot', 0):.2f}"),
                ("Fear & Greed", f"{snapshot.get('fear_greed_score', 0):.0f}"),
                ("Market Breadth", f"{snapshot.get('market_breadth', 0)*100:.1f}%"),
            ]
            
            if vrp_data:
                data_items.extend([
                    ("VRP", f"{vrp_data.get('vrp', 0):+.2f}"),
                    ("Realized Vol", f"{vrp_data.get('realized_vol', 0):.2f}"),
                    ("VIX Regime", vrp_data.get('regime', 'N/A')),
                ])
            
            # Draw table
            col_width = 90
            row_height = 7
            
            pdf.set_fill_color(240, 240, 240)
            
            for i, (label, value) in enumerate(data_items):
                fill = i % 2 == 0
                pdf.cell(col_width, row_height, label, 1, 0, 'L', fill)
                pdf.cell(col_width, row_height, str(value), 1, 1, 'R', fill)
        
        # Footer note
        pdf.ln(10)
        pdf.set_font('Arial', 'I', 8)
        pdf.set_text_color(128, 128, 128)
        pdf.multi_cell(0, 4,
            "Disclaimer: This report is for informational purposes only and does not constitute financial advice. "
            "Market conditions can change rapidly. Always conduct your own research and consult with a qualified "
            "financial advisor before making investment decisions."
        )
        
        # Save PDF
        pdf.output(str(output_path))
        logger.info(f"PDF report generated: {output_path}")
        
        return str(output_path)


def create_sample_report():
    """Create a sample report for testing"""
    generator = PDFReportGenerator()
    
    sample_snapshot = {
        'date': '2025-11-27',
        'credit_spread_hy': 3.10,
        'credit_spread_ig': 4.75,
        'treasury_10y': 4.25,
        'vix_spot': 17.1,
        'fear_greed_score': 18,
        'market_breadth': 0.909,
    }
    
    sample_vrp = {
        'vrp': 1.73,
        'realized_vol': 15.37,
        'regime': 'Elevated',
    }
    
    output = generator.generate_report(
        snapshot=sample_snapshot,
        vrp_data=sample_vrp,
        charts={},
        filename="sample_report.pdf"
    )
    
    return output
