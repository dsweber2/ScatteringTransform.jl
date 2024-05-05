function rotate_image(img, angle)
	
	if length(size(img)) == 4
		x, y, fea_size, sample_size = size(img);

		for fea = 1 : fea_size

			for m = 1 : sample_size

				rot_img = imrotate(img[:,:,fea,m], angle);
				rot_img[isnan.(rot_img)] .= 0;
				rot_img_par = parent(rot_img);
				if fea == 1 & m == 1
					xx, yy = size(rot_img_par);
					global img_rotated = zeros(xx, yy, fea_size, sample_size);
				end
				img_rotated[:, :, fea, m] = rot_img_par;
			end

		end

	elseif length(size(img)) == 5
		x, y, fea_size1, fea_size2, sample_size = size(img);

		for fea1 = 1 : fea_size1

			for fea2 = 1 : fea_size2

				for m = 1 : sample_size

					rot_img = imrotate(img[:,:,fea1,fea2,m], angle);
					rot_img[isnan.(rot_img)] .= 0;
					rot_img_par = parent(rot_img);
					if fea1 == 1 & fea2 == 1 & m == 1
						xx, yy = size(rot_img_par);
						global img_rotated = zeros(xx, yy, fea_size1, fea_size2, sample_size);
					end
					img_rotated[:, :, fea1, fea2, m] = rot_img_par;
				end

			end

		end

	end

	return img_rotated

end