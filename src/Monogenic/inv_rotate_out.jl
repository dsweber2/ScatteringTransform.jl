function inv_rotate_out(out, img, angle)

	# Input
	# out: the ouput after the scattering transform
	# img: the target image size for the scattering transform
	# angle: the rotation angle

	x, y = size(img);

	if length(size(out)) == 4

	    xx, yy, fea_out, sample_size = size(out);

	    half_x = Int(ceil(x / 2));
	    half_y = Int(ceil(y / 2));

	    if isodd(x) == 1
	        odd_x = 1;
	    else
	        odd_x = 0;
	    end

	    if isodd(y) == 1
	        odd_y = 1;
	    else
	        odd_y = 0;
	    end

		global out_new = zeros(x, y, fea_out, sample_size);

		for m = 1:sample_size
			for kk = 1:fea_out
			    back_img = imrotate(out[:,:,kk,m], angle, axes(out[:,:,kk,m]));
			    xx, yy = size(back_img);
			    center_x = Int(floor(xx / 2));
	                    center_y = Int(floor(yy / 2));

			    back_img[isnan.(back_img)] .= 0;
			    back_img = parent(back_img);
			    back_img = back_img[(center_x - half_x + 1 + odd_x): (center_x + half_x),
				(center_y - half_y + 1 + odd_y) : center_y + half_y];
			    
			    global out_new[:,:,kk,m] = back_img;
			end
		end

		return out_new


	elseif length(size(out)) == 5

	    xx, yy, fea_out, fea_size, sample_size = size(out);

	    half_x = Int(ceil(x / 2));
	    half_y = Int(ceil(y / 2));

		if isodd(x) == 1
	        odd_x = 1;
	    else
	        odd_x = 0;
	    end

	    if isodd(y) == 1
	        odd_y = 1;
	    else
	        odd_y = 0;
	    end

		global out_new = zeros(x, y, fea_out, fea_size, sample_size);

		for fea = 1:fea_size
			for m = 1:sample_size
				for kk = 1:fea_out
				    back_img = imrotate(out[:,:,kk,fea,m], angle, axes(out[:,:,kk,fea,m]));
			            xx, yy = size(back_img);
			            center_x = Int(floor(xx / 2));
	                            center_y = Int(floor(yy / 2));

				    back_img[isnan.(back_img)] .= 0;
				    back_img = parent(back_img);
				    back_img = back_img[(center_x - half_x + 1 + odd_x): (center_x + half_x),
					(center_y - half_y + 1 + odd_y) : center_y + half_y];
				    
				    global out_new[:,:,kk,fea,m] = back_img;
				end
			end
		end

		return out_new

	elseif length(size(out)) == 6

		xx, yy, fea_out1, fea_out2, fea_size, sample_size = size(out);

		half_x = Int(ceil(x / 2));
		half_y = Int(ceil(y / 2));

		if isodd(x) == 1
		    odd_x = 1;
		else
		    odd_x = 0;
		end

		if isodd(y) == 1
		    odd_y = 1;
		else
		    odd_y = 0;
		end

		global out_new = zeros(x, y, fea_out1, fea_out2, fea_size, sample_size);

		for fea = 1:fea_size
			for m = 1:sample_size
				for jj = 1:fea_out1
					for kk = 1:fea_out2
					    back_img = imrotate(out[:,:,jj,kk,fea,m], angle, axes(out[:,:,jj,kk,fea,m]));
					    xx, yy = size(back_img);
					    center_x = Int(floor(xx / 2));
					    center_y = Int(floor(yy / 2));
	
					    back_img[isnan.(back_img)] .= 0;
					    back_img = parent(back_img);
					    back_img = back_img[(center_x - half_x + 1 + odd_x): (center_x + half_x),
						(center_y - half_y + 1 + odd_y) : center_y + half_y];
					    
					    global out_new[:,:,jj,kk,fea,m] = back_img;
					end
				end
			end
		end

		return out_new

	else 
		print("The dimension of the ST output is ");
		print(length(size(out)));
		print("\n");
	end



	

end
