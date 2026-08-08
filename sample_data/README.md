Put test images here, e.g.:

- `receipt.jpg` — a photo of a Bangladeshi retail receipt/invoice
- `product.jpg` — a photo of a physical product

Test with:
```
curl -X POST http://localhost:8000/analyze -F "image=@sample_data/receipt.jpg"
```
