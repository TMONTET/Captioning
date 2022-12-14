import {Injectable} from "@angular/core";
import {HttpClient, HttpHeaders} from "@angular/common/http";
import {map, Observable, tap} from "rxjs";
import {ImageCaption, ImageVerification} from "./types";

@Injectable()
export class KerasService {

  constructor(private httpClient: HttpClient) {}

  public verify(file: File): Observable<ImageVerification> {
    let formData: FormData = new FormData();
    formData.append('file', file);

    return this.httpClient.post('http://127.0.0.1:5000/verify', formData).pipe(
      map(result => result as ImageVerification));
  }

  public noise(file: File): Observable<any> {
    let formData: FormData = new FormData();
    formData.append('file', file);

    return this.httpClient.post('http://127.0.0.1:5000/noise', formData, {responseType: 'blob'});
  }

  public denoise(file: File): Observable<any> {
    let formData: FormData = new FormData();
    formData.append('file', file);

    return this.httpClient.post('http://127.0.0.1:5000/denoise', formData, {responseType: 'blob'});
  }

  public caption(file: File): Observable<ImageCaption> {
    let formData: FormData = new FormData();
    formData.append('file', file);

    return this.httpClient.post('http://127.0.0.1:5000/caption', formData).pipe(
      map(result => result as ImageCaption));
  }

}
