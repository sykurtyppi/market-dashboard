"""
Comprehensive PDF Report Generator
Creates 5-page professional market analysis report
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
import io

logger = logging.getLogger(__name__)


class ComprehensivePDFReport:
    """Generate comprehensive 5-page market report"""
    
    def __init__(self):
        self.width, self.height = letter
        
    def generate(self, snapshot: dict, charts: dict, output_filename: str = "market_report.pdf"):
        """
        Generate 5-page PDF report
        
        Args:
            snapshot: Latest market data from database
            charts: Dict of Plotly figure objects
            output_filename: Name of output file
        
        Returns:
            Path to generated PDF
        """
        output_path = f"/mnt/user-data/outputs/{output_filename}"
        c = canvas.Canvas(output_path, pagesize=letter)
        
        # Generate each page
        self._page1_executive_summary(c, snapshot)
        self._page2_vix_term_structure(c, snapshot, charts)
        self._page3_breadth_analysis(c, snapshot, charts)
        self._page4_credit_liquidity(c, snapshot, charts)
        self._page5_treasury_stress(c, snapshot, charts)
        
        # Save
        c.save()
        logger.info(f"Generated comprehensive PDF: {output_path}")
        
        return output_path
    
    def _page1_executive_summary(self, c, snapshot):
        """Page 1: Executive Summary with regime and key metrics"""
        
        # Header
        c.setFont("Helvetica-Bold", 24)
        c.drawString(1*inch, self.height - 1*inch, "Market Risk Dashboard")
        
        c.setFont("Helvetica", 12)
        now = datetime.now().strftime('%Y-%m-%d %H:%M')
        c.drawString(1*inch, self.height - 1.3*inch, f"Report Generated: {now}")
        c.drawString(1*inch, self.height - 1.5*inch, f"Data Updated: {snapshot.get('updated_at', 'N/A')}")
        
        # Market Regime Section
        y_pos = self.height - 2.2*inch
        c.setFont("Helvetica-Bold", 16)
        c.drawString(1*inch, y_pos, "Market Regime Summary")
        
        y_pos -= 0.4*inch
        c.setFont("Helvetica", 11)
        
        # Build regime string
        regimes = []
        
        if snapshot.get('credit_regime'):
            regimes.append(f"Credit: {snapshot['credit_regime']}")
        
        if snapshot.get('vol_regime'):
            regimes.append(f"Volatility: {snapshot['vol_regime']}")
        
        if snapshot.get('vrp_regime'):
            regimes.append(f"VRP: {snapshot['vrp_regime']}")
        
        if snapshot.get('sentiment_regime'):
            regimes.append(f"Sentiment: {snapshot['sentiment_regime']}")
        
        for regime in regimes:
            c.drawString(1.2*inch, y_pos, f"• {regime}")
            y_pos -= 0.25*inch
        
        # Key Metrics Table
        y_pos -= 0.3*inch
        c.setFont("Helvetica-Bold", 16)
        c.drawString(1*inch, y_pos, "Key Market Indicators")
        
        y_pos -= 0.5*inch
        
        # Prepare metrics data
        metrics_data = [
            ['Indicator', 'Value', 'Status'],
            ['VIX Spot', f"{snapshot.get('vix', 0):.2f}", ''],
            ['VIX Contango', f"{snapshot.get('vix_contango', 0):+.1f}%", snapshot.get('vol_regime', '')],
            ['HY Spread', f"{snapshot.get('credit_spread_hy', 0):.0f} bps", ''],
            ['IG Spread', f"{snapshot.get('credit_spread_ig', 0):.0f} bps", ''],
            ['Market Breadth', f"{snapshot.get('market_breadth', 0)*100:.1f}%", ''],
            ['Fear & Greed', f"{snapshot.get('fear_greed', 0):.0f}", snapshot.get('sentiment_regime', '')],
            ['10Y Treasury', f"{snapshot.get('treasury_10y', 0):.2f}%", ''],
        ]
        
        table = Table(metrics_data, colWidths=[2.5*inch, 1.5*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
        ]))
        
        table.wrapOn(c, self.width, self.height)
        table.drawOn(c, 1*inch, y_pos - 2.5*inch)
        
        # Footer
        c.setFont("Helvetica-Oblique", 9)
        c.setFillColorRGB(0.5, 0.5, 0.5)
        c.drawString(1*inch, 0.5*inch, "Market Risk Dashboard | Not financial advice | Data: FRED, CBOE, Yahoo Finance, Fed Treasury")
        c.setFillColorRGB(0, 0, 0)
    
    def _page2_vix_term_structure(self, c, snapshot, charts):
        """Page 2: VIX Term Structure"""
        c.showPage()
        
        # Header
        c.setFont("Helvetica-Bold", 22)
        c.drawString(1*inch, self.height - 1*inch, "VIX Term Structure")
        
        c.setFont("Helvetica", 11)
        c.drawString(1*inch, self.height - 1.3*inch, "Volatility expectations across time horizons")
        
        # Chart
        if 'VIX_Term_Structure' in charts:
            try:
                img_buffer = io.BytesIO()
                charts['VIX_Term_Structure'].write_image(img_buffer, format='png', width=700, height=400)
                img_buffer.seek(0)
                c.drawImage(img_buffer, 0.75*inch, self.height - 5.5*inch, 
                           width=6.5*inch, height=3.5*inch, preserveAspectRatio=True, mask='auto')
            except Exception as e:
                logger.error(f"Error adding VIX chart: {e}")
                c.setFont("Helvetica", 12)
                c.drawString(1*inch, self.height - 3*inch, "VIX Term Structure chart unavailable")
        
        # Current values
        y_pos = self.height - 6.2*inch
        c.setFont("Helvetica-Bold", 14)
        c.drawString(1*inch, y_pos, "Current Values:")
        
        y_pos -= 0.3*inch
        c.setFont("Helvetica", 11)
        c.drawString(1.2*inch, y_pos, f"VIX (30d): {snapshot.get('vix', 0):.2f}")
        y_pos -= 0.25*inch
        c.drawString(1.2*inch, y_pos, f"VIX3M (93d): {snapshot.get('vix3m', 0):.2f}")
        y_pos -= 0.25*inch
        c.drawString(1.2*inch, y_pos, f"Contango: {snapshot.get('vix_contango', 0):+.1f}%")
        
        # Interpretation
        y_pos -= 0.5*inch
        c.setFont("Helvetica-Bold", 14)
        c.drawString(1*inch, y_pos, "Interpretation:")
        
        y_pos -= 0.3*inch
        c.setFont("Helvetica", 10)
        c.drawString(1.2*inch, y_pos, "• Contango (upward slope) = Normal conditions, calm markets")
        y_pos -= 0.22*inch
        c.drawString(1.2*inch, y_pos, "• Backwardation (downward slope) = Stress, elevated near-term fear")
        y_pos -= 0.22*inch
        c.drawString(1.2*inch, y_pos, "• Steep contango = Complacency, potential vol selling opportunity")
        y_pos -= 0.22*inch
        c.drawString(1.2*inch, y_pos, "• Flat curve = Transition period, uncertainty about future volatility")
        
        # Footer
        c.setFont("Helvetica-Oblique", 9)
        c.setFillColorRGB(0.5, 0.5, 0.5)
        c.drawString(1*inch, 0.5*inch, "Market Risk Dashboard | Page 2 | VIX Term Structure")
        c.setFillColorRGB(0, 0, 0)
    
    def _page3_breadth_analysis(self, c, snapshot, charts):
        """Page 3: Market Breadth"""
        c.showPage()
        
        # Header
        c.setFont("Helvetica-Bold", 22)
        c.drawString(1*inch, self.height - 1*inch, "Market Breadth Analysis")
        
        c.setFont("Helvetica", 11)
        c.drawString(1*inch, self.height - 1.3*inch, "Internal market participation and momentum")
        
        # Current signals
        y_pos = self.height - 1.8*inch
        c.setFont("Helvetica-Bold", 14)
        c.drawString(1*inch, y_pos, "Current Signals:")
        
        y_pos -= 0.3*inch
        c.setFont("Helvetica", 11)
        
        breadth_pct = snapshot.get('market_breadth', 0) * 100
        adv = snapshot.get('advancing_stocks', 0)
        dec = snapshot.get('declining_stocks', 0)
        c.drawString(1.2*inch, y_pos, f"Breadth: {breadth_pct:.1f}% ({adv}/{adv+dec} advancing)")
        
        y_pos -= 0.22*inch
        ad_ratio = adv / dec if dec > 0 else float('inf')
        c.drawString(1.2*inch, y_pos, f"A/D Ratio: {ad_ratio:.2f}x")
        
        y_pos -= 0.22*inch
        zweig_status = "ACTIVE ✓" if snapshot.get('zweig_thrust_active') else "None"
        c.drawString(1.2*inch, y_pos, f"Zweig Thrust: {zweig_status}")
        
        y_pos -= 0.22*inch
        div_type = snapshot.get('divergence_type', 'none').upper()
        c.drawString(1.2*inch, y_pos, f"Divergence: {div_type}")
        
        # A/D Line Chart
        if 'AD_Line' in charts:
            try:
                img_buffer = io.BytesIO()
                charts['AD_Line'].write_image(img_buffer, format='png', width=700, height=320)
                img_buffer.seek(0)
                c.drawImage(img_buffer, 0.75*inch, self.height - 5*inch, 
                           width=6.5*inch, height=2.8*inch, preserveAspectRatio=True, mask='auto')
            except Exception as e:
                logger.error(f"Error adding A/D Line chart: {e}")
        
        # McClellan Chart
        if 'McClellan' in charts:
            try:
                img_buffer = io.BytesIO()
                charts['McClellan'].write_image(img_buffer, format='png', width=700, height=280)
                img_buffer.seek(0)
                c.drawImage(img_buffer, 0.75*inch, self.height - 8*inch, 
                           width=6.5*inch, height=2.5*inch, preserveAspectRatio=True, mask='auto')
            except Exception as e:
                logger.error(f"Error adding McClellan chart: {e}")
        
        # Footer
        c.setFont("Helvetica-Oblique", 9)
        c.setFillColorRGB(0.5, 0.5, 0.5)
        c.drawString(1*inch, 0.5*inch, "Market Risk Dashboard | Page 3 | Breadth Analysis")
        c.setFillColorRGB(0, 0, 0)
    
    def _page4_credit_liquidity(self, c, snapshot, charts):
        """Page 4: Credit & Liquidity"""
        c.showPage()
        
        # Header
        c.setFont("Helvetica-Bold", 22)
        c.drawString(1*inch, self.height - 1*inch, "Credit & Liquidity Conditions")
        
        # Credit Spreads
        c.setFont("Helvetica-Bold", 16)
        c.drawString(1*inch, self.height - 1.5*inch, "Credit Spreads")
        
        if 'Credit_Spreads' in charts:
            try:
                img_buffer = io.BytesIO()
                charts['Credit_Spreads'].write_image(img_buffer, format='png', width=700, height=320)
                img_buffer.seek(0)
                c.drawImage(img_buffer, 0.75*inch, self.height - 4.5*inch, 
                           width=6.5*inch, height=2.8*inch, preserveAspectRatio=True, mask='auto')
            except Exception as e:
                logger.error(f"Error adding credit chart: {e}")
        
        # Net Liquidity
        c.setFont("Helvetica-Bold", 16)
        c.drawString(1*inch, self.height - 5*inch, "Net Liquidity (RRP + TGA)")
        
        if 'Net_Liquidity' in charts:
            try:
                img_buffer = io.BytesIO()
                charts['Net_Liquidity'].write_image(img_buffer, format='png', width=700, height=320)
                img_buffer.seek(0)
                c.drawImage(img_buffer, 0.75*inch, self.height - 8.3*inch, 
                           width=6.5*inch, height=2.8*inch, preserveAspectRatio=True, mask='auto')
            except Exception as e:
                logger.error(f"Error adding liquidity chart: {e}")
        
        # Footer
        c.setFont("Helvetica-Oblique", 9)
        c.setFillColorRGB(0.5, 0.5, 0.5)
        c.drawString(1*inch, 0.5*inch, "Market Risk Dashboard | Page 4 | Credit & Liquidity")
        c.setFillColorRGB(0, 0, 0)
    
    def _page5_treasury_stress(self, c, snapshot, charts):
        """Page 5: Treasury Stress"""
        c.showPage()
        
        # Header
        c.setFont("Helvetica-Bold", 22)
        c.drawString(1*inch, self.height - 1*inch, "Treasury Market Stress")
        
        c.setFont("Helvetica", 11)
        c.drawString(1*inch, self.height - 1.3*inch, "Bond market volatility and stress indicators")
        
        # MOVE Index Chart
        if 'Treasury_Stress' in charts:
            try:
                img_buffer = io.BytesIO()
                charts['Treasury_Stress'].write_image(img_buffer, format='png', width=700, height=350)
                img_buffer.seek(0)
                c.drawImage(img_buffer, 0.75*inch, self.height - 5*inch, 
                           width=6.5*inch, height=3*inch, preserveAspectRatio=True, mask='auto')
            except Exception as e:
                logger.error(f"Error adding MOVE chart: {e}")
        
        # Current values
        y_pos = self.height - 5.5*inch
        c.setFont("Helvetica-Bold", 14)
        c.drawString(1*inch, y_pos, "Current Levels:")
        
        y_pos -= 0.3*inch
        c.setFont("Helvetica", 11)
        move_value = snapshot.get('move_index', 0)
        c.drawString(1.2*inch, y_pos, f"MOVE Index: {move_value:.1f}")
        
        y_pos -= 0.25*inch
        sofr_rate = snapshot.get('sofr_rate', 0)
        c.drawString(1.2*inch, y_pos, f"SOFR Rate: {sofr_rate:.2f}%")
        
        y_pos -= 0.25*inch
        treasury_10y = snapshot.get('treasury_10y', 0)
        c.drawString(1.2*inch, y_pos, f"10Y Treasury: {treasury_10y:.2f}%")
        
        # Interpretation
        y_pos -= 0.5*inch
        c.setFont("Helvetica-Bold", 14)
        c.drawString(1*inch, y_pos, "Stress Level Guide:")
        
        y_pos -= 0.3*inch
        c.setFont("Helvetica", 10)
        c.drawString(1.2*inch, y_pos, "• < 80: Low stress, normal Treasury market conditions")
        y_pos -= 0.22*inch
        c.drawString(1.2*inch, y_pos, "• 80-100: Moderate stress, elevated uncertainty")
        y_pos -= 0.22*inch
        c.drawString(1.2*inch, y_pos, "• 100-150: Elevated stress, caution warranted")
        y_pos -= 0.22*inch
        c.drawString(1.2*inch, y_pos, "• > 150: High stress, potential systemic concerns")
        
        # Footer
        c.setFont("Helvetica-Oblique", 9)
        c.setFillColorRGB(0.5, 0.5, 0.5)
        c.drawString(1*inch, 0.5*inch, "Market Risk Dashboard | Page 5 | Treasury Stress | End of Report")
        c.setFillColorRGB(0, 0, 0)


def generate_comprehensive_pdf(snapshot: dict, charts: dict, filename: str = "market_report.pdf") -> str:
    """
    Convenience function to generate PDF
    
    Args:
        snapshot: Market data snapshot
        charts: Dictionary of Plotly figures
        filename: Output filename
    
    Returns:
        Path to generated PDF
    """
    generator = ComprehensivePDFReport()
    return generator.generate(snapshot, charts, filename)


if __name__ == "__main__":
    # Test with dummy data
    test_snapshot = {
        'updated_at': '2025-11-27',
        'credit_regime': 'Supportive',
        'vol_regime': 'Elevated',
        'vrp_regime': 'Positive',
        'sentiment_regime': 'Extreme Fear',
        'vix': 17.21,
        'vix_contango': 16.33,
        'vix3m': 20.02,
        'credit_spread_hy': 310,
        'credit_spread_ig': 475,
        'market_breadth': 0.69,
        'fear_greed': 18,
        'treasury_10y': 4.25,
        'advancing_stocks': 69,
        'declining_stocks': 31,
        'zweig_thrust_active': False,
        'divergence_type': 'none',
        'move_index': 95.3,
        'sofr_rate': 4.55,
    }
    
    pdf_path = generate_comprehensive_pdf(test_snapshot, {}, "test_comprehensive_report.pdf")
    print(f"✅ Test PDF generated: {pdf_path}")
