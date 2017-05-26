from sky import ChromaticitySkyModel, get_seville_observer
from datetime import datetime, timedelta
from sys import argv
from ephem import city, Sun
import matplotlib.pyplot as plt
plt.ion()


sun = Sun()

if __name__ == '__main__':

    try:
        tau = float(argv[argv.index('-t')+1])
    except ValueError:
        tau = -1
    try:
        nside = int(argv[argv.index('-n')+1])
    except ValueError:
        nside = 32
    try:
        city_name = argv[argv.index('-p')+1]
    except ValueError:
        city_name = "Seville"
    if "seville" in city_name.lower():
        obs = get_seville_observer()
    else:
        obs = city(city_name)
    try:
        obs.date = datetime.strptime(argv[argv.index('-d')+1], "%Y-%m-%d")
    except ValueError:
        obs.date = datetime.now()
    try:
        delta = timedelta(minutes=int(argv[argv.index('-td')+1]))
    except ValueError:
        delta = timedelta(minutes=30)
    try:
        mode = ["luminance", "dop", "aop"].index(argv[argv.index('-m')+1].lower())
    except ValueError:
        mode = -1

    cur = obs.next_rising(sun).datetime()
    end = obs.next_setting(sun).datetime()
    if cur > end:
        cur = obs.previous_rising(sun).datetime()

    print "Simulating..."
    print "City: %s" % city_name, "- Turbidity: %.2f" % tau
    print "Date: %s" % obs.date.datetime().strftime("%Y-%m-%d")
    print "Mode: %s" % ["luminance", "dop", "aop"][mode], " - nside = %d" % nside

    plt.figure(1, figsize=(15, 15))
    while cur <= end:
        obs.date = cur
        sky = ChromaticitySkyModel(observer=obs, turbidity=tau, nside=nside)
        sky.generate()

        if mode == 0:
            ChromaticitySkyModel.plot_luminance(sky, fig=1, title="Luminance", mode="00100", sub=(1, 1, 1))
        elif mode == 1:
            ChromaticitySkyModel.plot_polarisation(sky, fig=1, title="", mode="10", sub=(1, 1, 1))
        elif mode == 2:
            ChromaticitySkyModel.plot_polarisation(sky, fig=1, title="", mode="01", sub=(1, 1, 1))

        plt.draw()
        plt.pause(.01)

        # increase the current time
        cur = cur + delta
    plt.show()
