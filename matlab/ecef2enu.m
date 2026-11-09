function ENU = ecef2enu(dXYZ, lat0_deg, lon0_deg)
    lat0 = deg2rad(lat0_deg);
    lon0 = deg2rad(lon0_deg);

    M = [ -sin(lon0)            ,  cos(lon0)            , 0
          -sin(lat0)*cos(lon0)  , -sin(lat0)*sin(lon0)  , cos(lat0)
           cos(lat0)*cos(lon0)  ,  cos(lat0)*sin(lon0)  , sin(lat0) ];

    ENU = (M * dXYZ')';
end
