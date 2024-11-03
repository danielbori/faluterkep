import webbrowser

e = iface.mapCanvas().extent()
xmax = e.xMaximum()
ymax = e.yMaximum()
xmin = e.xMinimum()
ymin = e.yMinimum()

firstsurvey = (f"https://maps.arcanum.com/hu/map/firstsurvey-hungary/"
            f"?bbox={xmin}%2C{ymin}%2C{xmax}%2C{ymax}")

secondsurvey = (f"https://maps.arcanum.com/hu/map/secondsurvey-hungary/"
            f"?bbox={xmin}%2C{ymin}%2C{xmax}%2C{ymax}")

cadastral = (f"https://maps.arcanum.com/hu/map/cadastral/"
            f"?layers=3%2C4&bbox={xmin}%2C{ymin}%2C{xmax}%2C{ymax}")


webbrowser.open(habsburg_kataszteri)