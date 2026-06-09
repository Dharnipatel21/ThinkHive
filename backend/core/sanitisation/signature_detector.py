import cv2
import numpy as np
import os


class SignatureDetector:
    """
    Detects and removes handwritten signatures from scanned
    document images using OpenCV contour detection.
    """

    def __init__(self):
        # Contour area thresholds for signature detection
        self.min_contour_area = 1000
        self.max_contour_area = 50000
        # Signatures typically appear in the bottom 30% of a page
        self.signature_region_threshold = 0.70

    def detect_signatures(self, image_path: str) -> dict:
        """
        Detect signature regions in a document image.

        Args:
            image_path: path to image file

        Returns:
            dict with has_signature, regions list, confidence
        """
        if not os.path.exists(image_path):
            return {"has_signature": False, "regions": [], "confidence": 0.0}

        try:
            # Read and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                return {"has_signature": False, "regions": [], "confidence": 0.0}

            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Threshold to binary
            _, thresh = cv2.threshold(
                gray, 0, 255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            # Find contours
            contours, _ = cv2.findContours(
                thresh,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            regions = []
            for contour in contours:
                area = cv2.contourArea(contour)

                # Filter by area range
                if not (self.min_contour_area < area < self.max_contour_area):
                    continue

                x, y, w, h = cv2.boundingRect(contour)

                # Check if in bottom portion of page (signature zone)
                if y / height < self.signature_region_threshold:
                    continue

                # Aspect ratio check - signatures are usually wider than tall
                aspect_ratio = w / h if h > 0 else 0
                if not (1.5 < aspect_ratio < 10):
                    continue

                regions.append({"x": x, "y": y, "width": w, "height": h})

            has_signature = len(regions) > 0
            confidence = min(1.0, len(regions) * 0.4) if has_signature else 0.0

            return {
                "has_signature": has_signature,
                "regions": regions,
                "confidence": round(confidence, 2)
            }

        except Exception as e:
            print(f"Signature detection error: {e}")
            return {"has_signature": False, "regions": [], "confidence": 0.0}

    def remove_signatures(self, image_path: str, output_path: str) -> str:
        """
        Black out detected signature regions in an image.

        Args:
            image_path: path to original image
            output_path: path to save cleaned image

        Returns:
            output_path of cleaned image
        """
        detection = self.detect_signatures(image_path)

        if not detection["has_signature"]:
            return image_path

        try:
            image = cv2.imread(image_path)

            for region in detection["regions"]:
                x = region["x"]
                y = region["y"]
                w = region["width"]
                h = region["height"]
                # Draw filled black rectangle over signature region
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), -1)

            cv2.imwrite(output_path, image)
            return output_path

        except Exception as e:
            print(f"Signature removal error: {e}")
            return image_path
