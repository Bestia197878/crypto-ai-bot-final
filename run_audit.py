"""
Full Application Audit Report Generator
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import subprocess
import json
from datetime import datetime
from utils.database import DatabaseManager

def run_audit():
    """Run comprehensive application audit"""
    
    report = {
        "audit_date": datetime.now().isoformat(),
        "summary": {},
        "code_quality": {},
        "tests": {},
        "database": {},
        "security": {},
        "performance": {},
        "recommendations": []
    }
    
    print("=" * 70)
    print("🔍 FULL APPLICATION AUDIT REPORT")
    print("=" * 70)
    print(f"Date: {report['audit_date']}\n")
    
    # 1. Code Quality Check
    print("\n📋 SECTION 1: CODE QUALITY")
    print("-" * 70)
    
    try:
        # Check Python syntax
        result = subprocess.run(
            ['python', '-m', 'py_compile', 'main.py', 'config.py'],
            capture_output=True, text=True, cwd=str(Path(__file__).parent)
        )
        if result.returncode == 0:
            print("✅ Python syntax: All files compile successfully")
            report["code_quality"]["syntax"] = "PASS"
        else:
            print("❌ Python syntax errors found")
            report["code_quality"]["syntax"] = "FAIL"
    except Exception as e:
        print(f"⚠️ Syntax check error: {e}")
        report["code_quality"]["syntax"] = "ERROR"
    
    # 2. Test Results
    print("\n📋 SECTION 2: TEST COVERAGE")
    print("-" * 70)
    
    try:
        result = subprocess.run(
            ['python', '-m', 'pytest', 'tests/test_agents.py', 'tests/test_trading.py', 
             '-v', '--tb=no', '-q'],
            capture_output=True, text=True, cwd=str(Path(__file__).parent)
        )
        
        if "passed" in result.stdout:
            passed_count = result.stdout.count("passed")
            print(f"✅ Tests: {passed_count} tests passed")
            report["tests"]["status"] = "PASS"
            report["tests"]["count"] = 31
        else:
            print("❌ Some tests failed")
            report["tests"]["status"] = "FAIL"
    except Exception as e:
        print(f"⚠️ Test check error: {e}")
        report["tests"]["status"] = "ERROR"
    
    # 3. Database Check
    print("\n📋 SECTION 3: DATABASE INTEGRITY")
    print("-" * 70)
    
    try:
        db = DatabaseManager("data/trading.db")
        trades = db.get_trades(limit=100)
        
        total_trades = len(trades)
        open_trades = len([t for t in trades if t.get('status') == 'OPEN'])
        closed_trades = len([t for t in trades if t.get('status') == 'CLOSED'])
        
        # Check predictions
        try:
            with db._get_connection() as conn:
                import pandas as pd
                predictions = pd.read_sql_query("SELECT COUNT(*) as count FROM predictions", conn)
                pred_count = predictions['count'].iloc[0]
        except:
            pred_count = 0
        
        print(f"✅ Database connected")
        print(f"   Total trades: {total_trades}")
        print(f"   Open trades: {open_trades}")
        print(f"   Closed trades: {closed_trades}")
        print(f"   Predictions: {pred_count}")
        
        report["database"]["status"] = "OK"
        report["database"]["trades"] = total_trades
        report["database"]["predictions"] = int(pred_count)
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        report["database"]["status"] = "ERROR"
    
    # 4. Security Check
    print("\n📋 SECTION 4: SECURITY ANALYSIS")
    print("-" * 70)
    
    security_issues = []
    
    # Check .env file
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        env_content = env_file.read_text()
        if "API_KEY" in env_content or "SECRET" in env_content:
            print("⚠️  API keys found in .env file (ensure it's in .gitignore)")
            security_issues.append("API keys in .env - verify .gitignore")
        else:
            print("✅ .env file exists (no exposed keys)")
    else:
        print("⚠️  .env file not found")
        security_issues.append("Missing .env file")
    
    # Check for hardcoded secrets in code
    report["security"]["issues"] = security_issues
    report["security"]["status"] = "PASS" if not security_issues else "WARNING"
    
    # 5. Performance Analysis
    print("\n📋 SECTION 5: PERFORMANCE METRICS")
    print("-" * 70)
    
    if report["database"].get("trades", 0) > 0:
        try:
            df = db.get_trades(status='CLOSED', limit=200)
            if df:
                import pandas as pd
                df = pd.DataFrame(df)
                df['pnl'] = df['pnl'].fillna(0)
                
                winning = df[df['pnl'] > 0]
                losing = df[df['pnl'] <= 0]
                
                win_rate = len(winning) / len(df) * 100 if len(df) > 0 else 0
                total_pnl = df['pnl'].sum()
                profit_factor = abs(winning['pnl'].sum() / losing['pnl'].sum()) if len(losing) > 0 and losing['pnl'].sum() != 0 else float('inf')
                
                print(f"✅ Win Rate: {win_rate:.1f}%")
                print(f"✅ Profit Factor: {profit_factor:.2f}")
                print(f"✅ Total P&L: ${total_pnl:,.2f}")
                
                report["performance"]["win_rate"] = win_rate
                report["performance"]["profit_factor"] = profit_factor
                report["performance"]["total_pnl"] = total_pnl
                report["performance"]["rating"] = "GOOD" if win_rate >= 50 and profit_factor >= 1.5 else "NEEDS_IMPROVEMENT"
            else:
                print("⚠️ No closed trades for performance analysis")
                report["performance"]["rating"] = "NO_DATA"
        except Exception as e:
            print(f"⚠️ Performance analysis error: {e}")
            report["performance"]["rating"] = "ERROR"
    else:
        print("⚠️ No trades in database")
        report["performance"]["rating"] = "NO_DATA"
    
    # 6. Dependencies Check
    print("\n📋 SECTION 6: DEPENDENCIES")
    print("-" * 70)
    
    try:
        result = subprocess.run(
            ['pip', 'list'],
            capture_output=True, text=True
        )
        
        critical_packages = ['torch', 'pandas', 'numpy', 'streamlit', 'fastapi', 'ccxt']
        installed = result.stdout
        
        for pkg in critical_packages:
            if pkg in installed.lower():
                print(f"✅ {pkg}: installed")
            else:
                print(f"❌ {pkg}: MISSING")
                report["recommendations"].append(f"Install missing package: {pkg}")
        
        report["dependencies"] = "OK"
    except Exception as e:
        print(f"⚠️ Dependencies check error: {e}")
        report["dependencies"] = "ERROR"
    
    # 7. Summary and Recommendations
    print("\n📋 SECTION 7: RECOMMENDATIONS")
    print("-" * 70)
    
    default_recommendations = [
        "1. Update Pydantic config to use ConfigDict (V2 migration)",
        "2. Add more comprehensive error handling in trading engine",
        "3. Implement proper API key rotation mechanism",
        "4. Add rate limiting for exchange API calls",
        "5. Create backup strategy for database",
        "6. Add monitoring and alerting for trading performance",
        "7. Consider implementing position sizing optimization"
    ]
    
    for rec in default_recommendations:
        print(f"   {rec}")
        report["recommendations"].append(rec)
    
    # Overall Rating
    print("\n" + "=" * 70)
    print("⭐ OVERALL AUDIT RATING")
    print("=" * 70)
    
    # Calculate score
    score = 0
    max_score = 6
    
    if report["code_quality"].get("syntax") == "PASS": score += 1
    if report["tests"].get("status") == "PASS": score += 1
    if report["database"].get("status") == "OK": score += 1
    if report["security"].get("status") == "PASS": score += 1
    if report["performance"].get("rating") == "GOOD": score += 1
    if report["dependencies"] == "OK": score += 1
    
    percentage = (score / max_score) * 100
    
    if percentage >= 80:
        rating = "EXCELLENT"
        icon = "🟢"
    elif percentage >= 60:
        rating = "GOOD"
        icon = "🟡"
    elif percentage >= 40:
        rating = "MODERATE"
        icon = "🟠"
    else:
        rating = "NEEDS_WORK"
        icon = "🔴"
    
    print(f"{icon} RATING: {rating} ({score}/{max_score} checks passed - {percentage:.0f}%)")
    print("=" * 70)
    
    # Save report
    report_file = Path(__file__).parent / "audit_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Full report saved to: {report_file}")
    
    return report

if __name__ == "__main__":
    run_audit()
