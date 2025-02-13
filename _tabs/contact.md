---
layout: page
title: Contact Us
icon: fas fa-envelope
order: 4
---

<form action="{{ site.contactform_url }}" method="POST" class="container">
  <div class="row mb-3">
    <div class="col-md-6">
      <label for="name" class="form-label">Name:</label>
      <input type="text" id="name" name="name" class="form-control" required>
    </div>
    <div class="col-md-6">
      <label for="email" class="form-label">Email:</label>
      <input type="email" id="email" name="email" class="form-control" required>
    </div>
  </div>

  <div class="row mb-3">
    <div class="col-12">
      <label for="message" class="form-label">Message:</label>
      <textarea id="message" name="message" class="form-control" rows="5" required></textarea>
    </div>
  </div>

  <div class="text-center">
    <button type="submit" style="background-color: #6c757d; color: white;">Send Message</button>
  </div>
</form>

