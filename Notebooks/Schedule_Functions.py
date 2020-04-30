import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta 

def countTrips(rSchedule):
    '''Given a schedule, returns total number of trips'''
    return len(rSchedule.trip_id.drop_duplicates())

def startTime(rSchedule):
    '''
    Given a schedule, returns the first departure time of a trip.
    '''
    rSchedule = rSchedule[rSchedule.trip_stop_sequence ==1]
    times = rSchedule.dep_time.apply(lambda x: datetime.strptime(x, '%I:%M %p') if type(x) == str else "")
    times = [t for t in times if t.hour >= 3]
    if len(times)>0:
        min_time = min(times).strftime('%H:%M:%S')
    else:
        min_time = "-"
    return min_time

# Get Peak Hours and oof Peak Hour parts of schedules
def peakHours(rSchedule, peak_ratio = 0.333,
             peak_min1 = 7, peak_max1 = 10,
             peak_min2 = 16, peak_max2 = 19):
    '''
    Given a schedule, returns two dataframes: one that includes peak trips and one that includes off peak trips.
    The dataframes also include "periods", which define discrete time chunks. For example, 7-10am is period 1
    of peak hours and 4-7pm is period 2 of peak hours.
    
    A trip is considered peak if the percentage of its stops that occur during peak hours is greater than or 
    equal to the peak_ratio.
    
    Peak hours are definied by parameter inputs as well with the first peak period occuring between peak_min1 & peak_max1
    and the next peak period between peak_min2 & peak_max2. All trips not classifed as peak are classified as off peak.
    '''
    
    peak_trips = pd.DataFrame(columns = ['Trip', 'Period'])
    off_peak_trips = pd.DataFrame(columns = ['Trip', 'Period'])
    
    rSchedule = rSchedule.reset_index(drop = True)
    
    trip_list = list(rSchedule.trip_id.unique())
    
    for trip in trip_list:
        
        temp_df = rSchedule[rSchedule.trip_id == trip].reset_index(drop = True)
        
        during_peak = list()
        
        for i in range(temp_df.shape[0]):
            
            if pd.isna(temp_df.arr_time[i]):
                continue
            
            temp_hr = datetime.strptime(temp_df.arr_time[i], '%I:%M %p').hour
            
            if temp_hr < peak_min1:
                off_peak_period = 1
                during_peak.append(0)
            elif temp_hr < peak_max1:
                peak_period = 1
                during_peak.append(1)
            elif temp_hr < peak_min2:
                off_peak_period = 2
                during_peak.append(0)
            elif temp_hr < peak_max2:
                peak_period = 2
                during_peak.append(1)
            else:
                off_peak_period = 3
                during_peak.append(0)
        
        if len(during_peak) > 0:
            percent_peak = sum(during_peak)/len(during_peak)
            
            if percent_peak >= peak_ratio:
                
                peak_trips = peak_trips.append(pd.DataFrame(zip([trip], [peak_period]),
                                                                   columns = ['Trip', 'Period']))
            else:
                off_peak_trips = off_peak_trips.append(pd.DataFrame(zip([trip], [off_peak_period]),
                                                                   columns = ['Trip', 'Period']))
        
    return (peak_trips, off_peak_trips)

def orderTrips(rSchedule):
    ''' Function that takes trips and orders them by the time they leave their first stop'''
    
    sched = rSchedule.copy()
    
    starts_df = sched[sched.trip_stop_sequence == 1].reset_index(drop = True)
    starts_df['dep_time'] = starts_df.dep_time.apply(lambda x: 
                                                    x if pd.isna(x) else datetime.strptime(x, '%I:%M %p'))
    trip_order = list(starts_df.sort_values('dep_time').trip_id)
    
    sched['trip_id'] = pd.Categorical(sched['trip_id'], trip_order)
    
    return sched.sort_values(['direction_id', 'trip_id', 'trip_stop_sequence']).reset_index(drop = True)

def avgHeadway(rSchedule, periodDF = None):
    '''
    Given a schedule, returns the average headway for all stops in that schedule. Also takes a periodDF object that gives
    a dataframe that defines the periods of different trips (output of peakHours function). If periodDF is not provided,
    all trips are taken to be as in the same period.
    '''
    
    sched = rSchedule.copy()
    
    sched = orderTrips(sched)
    
    # Convert arr_time to datettime type
    sched['arr_time'] = \
        sched.arr_time.apply(lambda x: x if pd.isna(x) else datetime.strptime(x, '%I:%M %p'))
    
    if not periodDF is None:
        # Combine dataframes
        sched = pd.merge(sched, periodDF, left_on="trip_id", right_on="Trip")
        periods = list(sched.Period.unique())
    
    time_difference = list()
    
    if not periodDF is None:
    
        for period in periods:
            
            period_df = sched[sched.Period == period].reset_index(drop = True)
            stops = list(period_df.stop_id.unique())
            
            for stop in stops:
                    
                stop_df = period_df[period_df.stop_id == stop].reset_index(drop = True)
                
                directions = list(stop_df.direction_id.unique())
                
                for d in directions:
                    
                    dir_df = stop_df[stop_df.direction_id == d].reset_index(drop = True)
                
                    ordered_arrivals = dir_df.arr_time.sort_values().reset_index(drop = True)
                
                    for i in range(len(ordered_arrivals)-1):

                        if (pd.notna(ordered_arrivals[i])) & (pd.notna(ordered_arrivals[i+1])):
                            
                            time_difference.append(ordered_arrivals[i+1] - ordered_arrivals[i])
    
    else:
        stops = list(sched.stop_id.unique())

        for stop in stops:
                    
            stop_df = sched[sched.stop_id == stop].reset_index(drop = True)

            directions = list(stop_df.direction_id.unique())

            for d in directions:

                dir_df = stop_df[stop_df.direction_id == d].reset_index(drop = True)

                ordered_arrivals = dir_df.arr_time.sort_values().reset_index(drop = True)

                for i in range(len(ordered_arrivals)-1):

                    if (pd.notna(ordered_arrivals[i])) & (pd.notna(ordered_arrivals[i+1])):

                        time_difference.append(ordered_arrivals[i+1] - ordered_arrivals[i])
                        
    time_sum = [d.seconds/60 for d in time_difference]
    
    if len(time_sum) == 0:
        return "-"
    else:
        return sum(time_sum)/len(time_sum)
    
def timeRange(rSchedule):
    '''
    Given a schedule, returns the average length of time between the first and last stops in trips in that schedule.
    '''
    
    if rSchedule.shape[0] == 0:
        return "-"
    
    sched = rSchedule.copy()
    
    times_df = pd.DataFrame(columns = ["Route", "Time"])
    
    times = list()
    
    trips = list(sched.trip_id.unique())
    
    for trip in trips:
        
        trip_df = sched[sched.trip_id == trip].sort_values("trip_stop_sequence").reset_index(drop = True)
        
        for i in range(trip_df.shape[0]):
            if pd.notna(trip_df.arr_time[i]):
                start_index = i
                break
            else:
                start_index = -1
        
        if start_index == -1:
            break
        else:
            for j in range(start_index + 1, trip_df.shape[0]):
                if pd.notna(trip_df.arr_time[j]):
                    end_index = j
        
        difference = datetime.strptime(trip_df.arr_time[end_index], '%I:%M %p') - \
                     datetime.strptime(trip_df.arr_time[start_index], '%I:%M %p')
        
        times.append(difference.seconds / 60)
        
    if len(times) == 0:
        return "-"
    else:
        return sum(times) / len(times)
    
def remove_complete_na_trips(schedule):
    '''
    Given a schedule, removes trips that have arrival times that are entirely NA.
    '''
    
    trips = schedule.trip_id.unique()
    trips_to_keep = list()
    
    for trip in trips:
        temp = schedule[schedule.trip_id == trip]
        
        last_stop = max(temp.trip_stop_sequence)
        
        if not (pd.isna(temp.arr_time[temp.trip_stop_sequence == last_stop].iloc[0]) and
                pd.isna(temp.dep_time[temp.trip_stop_sequence == 1].iloc[0])):
            
            trips_to_keep.append(trip)
            
    mask = [t in trips_to_keep for t in schedule.trip_id]
    
    return schedule[mask]

def peak_vehicle(schedule, recovery_factor = 0.0):
    '''
    Given a schedule, returns the peak number of vehicles required to fulfill that schedule.
    Recovery factor is the percent of the trip that must be spent recovering.
    '''
    
    ### return "-" for empty dataframes
    if schedule.shape[0] == 0:
        return "-"
    
    ## Remove trips that have no departure time & arrival time information
    schedule = remove_complete_na_trips(schedule)
    
    ## First order schedule by trips and stops
    schedule = schedule.sort_values(["trip_id", "trip_stop_sequence"]).reset_index(drop = True)
    
    ## Second go through all trips to get two lists. One with all departure times from first stops
    ## and second with (adjusted by recovery factor) arrival times for all last stops on trip.
    
    departure_times = list()
    arrival_times = list()

    for i in range(schedule.shape[0]):

        if i == 0:
            departure_times.append(datetime.strptime(schedule.dep_time[i], '%I:%M %p'))

        elif i == (schedule.shape[0]-1):

            if pd.isna(schedule.arr_time[i]):
                j = i-1
                while(pd.isna(schedule.arr_time[j])):
                    j = j-1
                arrival_times.append(datetime.strptime(schedule.arr_time[j], '%I:%M %p'))
            else:
                arrival_times.append(datetime.strptime(schedule.arr_time[i], '%I:%M %p'))

        elif schedule.trip_stop_sequence[i] == 1:

            # Some cases where we have na in columns
            if pd.isna(schedule.arr_time[i-1]):
                j = i-1
                while(pd.isna(schedule.arr_time[j])):
                    j = j-1
                arrival_times.append(datetime.strptime(schedule.arr_time[j], '%I:%M %p'))
            else:
                arrival_times.append(datetime.strptime(schedule.arr_time[i-1], '%I:%M %p'))

            departure_times.append(datetime.strptime(schedule.dep_time[i], '%I:%M %p'))

    # Accouting for recovery factor, we adjust arrival times
    adjusted_arrivals = [arrival_times[i] + ((arrival_times[i] - departure_times[i]) * (recovery_factor)) \
                         for i in range(len(arrival_times))]
    
    ## Third, create a dataframe that has all first and last stop times.
    trips_df = pd.DataFrame(zip(departure_times, adjusted_arrivals), columns = ["Departure", "Arrival"])

    # Assume that any trip that departs (or arrives) between the hours of 12am and 3am occurs on the next day
    for i in range(0, trips_df.shape[0]):
    
        if trips_df.Departure[i].hour <= 3:
            trips_df.loc[i, "Departure"] = trips_df.loc[i, "Departure"] + timedelta(days=1)
            trips_df.loc[i, "Arrival"] = trips_df.loc[i, "Arrival"] + timedelta(days=1)
        elif trips_df.Arrival[i].hour <= 3:
            trips_df.loc[i, "Arrival"] = trips_df.loc[i, "Arrival"] + timedelta(days=1)

    trips_df = trips_df.sort_values("Departure").reset_index(drop=True)
    
    ## Finally  count how many busses overlap at any one time and record the maximums
    peak_vehicles = 0

    for i in range(trips_df.shape[0]):

        if i != trips_df.shape[0]-1:
            inservice_vehicles = 0
            temp_arrival = trips_df.Arrival[i]

            j=i

            while trips_df.Departure[j] < temp_arrival:
                inservice_vehicles += 1
                j+=1

                if j == trips_df.shape[0]-1: break

            if inservice_vehicles > peak_vehicles:
                peak_vehicles = inservice_vehicles
                
    return peak_vehicles
