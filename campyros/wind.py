import getgfs, dateutil.parser, pickle, warnings
from datetime import datetime, date, timedelta
from pathlib import Path
import numpy as np

__copyright__ = """

    Copyright 2021 Jago Strong-Wright

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

Path("data/wind").mkdir(parents=True, exist_ok=True)


class Wind:
    def __init__(
        self,
        datetime=(date.today() - timedelta(days=2)).strftime("%Y%m%d %H:%M"),
        default=np.array([0, 0, 0]),
        variable=True,
        cache=False,
    ):
        """Initites wind object so wind for some position can be searched

        Note
        ----
        - Historic forcasts are not yet implimented in getgfs so historics not available in this

        Args:
            datetime (string): datetime of flight, can be in 'any' format as long as datetime parser can work it out (for now this will need to be UTC)
            cache (bool, optional): cache lat/long/time profile, useful for stats models where downloading for each run is a bottle neck. Defaults to False.
        """
        self.launch_time = dateutil.parser.parse(datetime).timestamp()
        self.cache = cache
        self.default = default
        self.variable = variable

        self.forecast = getgfs.Forecast("0p25", "1hr")
        self.profiles = {}  # (lat,long,datetime):interp profile
        self.points = []  # tuples (lat,long,forecast)

    def get_wind(self, lat, long, alt, flight_time=0):
        """[summary]

        Args:
            lat (float): latitude
            long (float): longitude
            alt (float): altitude
            flight_time (float, optional): time into flight. Defaults to 0.

        Returns:
            tuple: returns u and v components of wind
        """
        if self.variable == True:
            lat = [
                float(self.forecast.coords["lat"]["resolution"]) * n
                + float(self.forecast.coords["lat"]["minimum"])
                for n in range(0, int(self.forecast.coords["lat"]["grads_size"]))
            ][self.forecast.value_to_index("lat", lat)]
            long = [
                float(self.forecast.coords["lon"]["resolution"]) * n
                + float(self.forecast.coords["lon"]["minimum"])
                for n in range(0, int(self.forecast.coords["lon"]["grads_size"]))
            ][self.forecast.value_to_index("lon", long)]
            request_timestamp = self.launch_time + flight_time
            request_datetime = datetime.fromtimestamp(request_timestamp).strftime(
                "%Y-%m-%d %H:%M"
            )
            forecast_to_use = self.forecast.datetime_to_forecast(request_datetime)
            if (lat, long, forecast_to_use) not in self.points:
                if (
                    self.cache == False
                    or not Path(
                        "data/wind/%s_%s_%s.pkl" % (lat, long, forecast_to_use)
                    ).is_file()
                ):
                    try:
                        self.profiles[
                            (lat, long, forecast_to_use)
                        ] = self.forecast.get_windprofile(request_datetime, lat, long)
                        self.points.append((lat, long, forecast_to_use))
                    except Exception as e:
                        warnings.warn(
                            "Wind could not be found - defaults used, gfspy gave the message:"
                        )
                        print(e)
                        return self.default

                    if self.cache == True:
                        with open(
                            "data/wind/%s_%s_%s.pkl" % (lat, long, forecast_to_use),
                            "wb",
                        ) as dump_file:
                            pickle.dump(
                                self.profiles[(lat, long, forecast_to_use)], dump_file
                            )
                else:
                    with open(
                        "data/wind/%s_%s_%s.pkl" % (lat, long, forecast_to_use), "rb"
                    ) as dump_file:
                        self.profiles[(lat, long, forecast_to_use)] = pickle.load(
                            dump_file
                        )
                    self.points.append((lat, long, forecast_to_use))

            return np.array(
                [
                    self.profiles[(lat, long, forecast_to_use)][0](alt),
                    self.profiles[(lat, long, forecast_to_use)][1](alt),
                    0,
                ]
            )
        else:
            return self.default


if __name__ == "__main__":
    w = Wind("20210428 15:00", cache=True)
    print(w.get(2938, 29328, 20866566))
    """times=np.linspace(0,60*60*12,1000)
    x=[]
    for t in times:
        x.append(w.get(37,20,1000,t)[0])
    import matplotlib.pyplot as plt
    plt.plot(times,x)
    plt.show()"""
