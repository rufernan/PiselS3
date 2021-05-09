function angle = angle_range_positive (angle)
    angle = angle_range_positive_up   (angle);
    angle = angle_range_positive_down (angle);
end

function angle = angle_range_positive_up (angle)
    idx = (angle >= 360);
    if ~any(idx),  return;  end
    angle(idx) = mod(angle(idx), 360);
    angle = angle_range_positive_up (angle);
end

function angle = angle_range_positive_down (angle)
    idx = (angle < 0);
    if ~any(idx),  return;  end
    angle(idx) = 360 + angle(idx);
    angle = angle_range_positive_down (angle);
end

%!test
%! temp = [
%!     0       0
%!     180     180
%!     -180    +180
%!     -150    +210
%!     -360    0
%!     +370    +10
%!     -370    +350
%!     +730    +10
%!     -730    +350
%! ];
%! temp(:,3) = angle_range_positive (temp(:,1));
%! %temp  % DEBUG
%! myassert(temp(:,3), temp(:,2), -sqrt(eps()))

%!test
%! n = 10;
%! angle1 = randint(-1000, +1000, n, 1);
%! angle3 = angle_range_positive(angle1);  % (requires iteration)
%! angle2 = angle(exp(1i*angle1*pi/180))*180/pi;
%! angle4 = angle_range_positive(angle2);  % doesn't require iteration
%! %[angle1, angle2, angle3, angle4, angle4-angle3]  % DEBUG
%! myassert(angle4, angle3, -sqrt(eps()))
